import os
import torch
import tqdm
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from groundeddino_vl import load_model, predict


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="models/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--chkpot_path", default="weights/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--input_dir", default="fake_images", help="folder containing images")
    parser.add_argument("--txt_output_dir", default="fake_labels", help="where to save voc style labels")
    parser.add_argument("--img_output_dir", default="fake_detections", help="where to save visualized images")
    parser.add_argument("--save_txt", type=bool, default=True, help="whether to save the VOC label result")
    parser.add_argument("--save_img", type=bool, default=False, help="whether to save the detection result")
    parser.add_argument("--text_prompt", default="construction site . building")
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_exist", action="store_true", help="skip if output image already exist")
    return parser.parse_args()


def gen_predictions(img_path, result, font=None):
    """
    get bounding box and draw on the image if needed
    """
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    if font is None:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    bounding_boxed = []

    for label, score, box in zip(result.labels, result.scores, result.boxes):
        # pass the label `building`
        if label == "building":
            continue

        x_center, y_center, width, height = box
        x_center *= w
        y_center *= h
        width *= w
        height *= h
        xmin = torch.clamp(x_center - width / 2, min=0)
        ymin = torch.clamp(y_center - height / 2, min=0)
        xmax = torch.clamp(x_center + width / 2, max=w)
        ymax = torch.clamp(y_center + height / 2, max=h)

        bounding_boxed.append([int(xmin.item()), int(ymin.item()), int(xmax.item()), int(ymax.item()), 0])

        label_text = f"{label}: {score:.2f}"
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        # text background
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        draw.rectangle([xmin, ymin - th, xmin + tw, ymin], fill="red")
        draw.text((xmin, ymin - th), label_text, fill="white", font=font)
    
    return image, bounding_boxed


def main():
    args = get_args()
    if args.save_txt:
        os.makedirs(args.txt_output_dir, exist_ok=True)
    if args.save_img:
        os.makedirs(args.img_output_dir, exist_ok=True)

    # load model once
    model = load_model(
        config_path=args.config_path,
        checkpoint_path=args.chkpot_path,
        device=args.device,
    )

    # collect all images
    img_ext = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    img_paths = [p for p in Path(args.input_dir).iterdir()
                 if p.suffix in img_ext]
    if not img_paths:
        print(f"No images found in {args.input_dir}")
        return

    # inference and save result
    for img_path in tqdm.tqdm(img_paths, desc="Processing"):
        img_save_path = Path(args.img_output_dir) / f"{img_path.stem}.jpg"
        txt_save_path = Path(args.txt_output_dir) / f"{img_path.stem}.txt"

        if args.skip_exist and img_save_path.exists():
            continue
        if args.skip_exist and txt_save_path.exists():
            continue

        result = predict(
            model=model,
            image=str(img_path),
            text_prompt=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        vis, bdx = gen_predictions(img_path, result)

        if args.save_img:
            vis.save(img_save_path)
        if args.save_txt:
            if not bdx:
                print(f"Detect nothing and delete the origin image: {str(img_path)}...")
                if img_path.exists():
                    img_path.unlink()
            else:
                with txt_save_path.open('w', encoding='utf-8') as f:
                    for row in bdx:
                        line = ','.join(map(str, row))
                        f.write(line + '\n')
    
    if args.save_txt:
        print(f"Done! Results saved to {args.txt_output_dir}")
    if args.save_img:
        print(f"Done! Results saved to {args.img_output_dir}")


if __name__ == "__main__":
    main()