import torch
import torch.nn as nn
import numpy as np
import os

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from torchvision import transforms
from torchvision import models

# Ensure this is available in your directory
from disease_info import disease_info

# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using Device: {device}")

# =====================================
# Load Classes
# =====================================

class_names = np.load(
    "models/class_names.npy",
    allow_pickle=True
)

# =====================================
# Load Model
# =====================================

model = models.mobilenet_v2()

model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(
        model.last_channel,
        len(class_names)
    )
)

model.load_state_dict(
    torch.load(
        "models/best_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# =====================================
# Image Transform
# =====================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# Select Image
# =====================================

image_path = "test.jpg"

image = Image.open(
    image_path
).convert("RGB")

image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

# =====================================
# Predict
# =====================================

with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, prediction = torch.max(probs, 1)

predicted_class = class_names[prediction.item()]

print("\nDisease:")
print(predicted_class)

print(
    "\nConfidence:",
    round(confidence.item()*100, 2),
    "%"
)

if predicted_class in disease_info:
    print("\nDisease Information:")
    print(disease_info[predicted_class])

# =====================================
# Save Prediction Chart Image
# =====================================

top_k = min(5, len(class_names))
top_probs, top_indices = torch.topk(probs[0], k=top_k)

top_probs = top_probs.cpu().numpy()
top_indices = top_indices.cpu().numpy()

os.makedirs("outputs", exist_ok=True)
chart_path = "outputs/prediction_chart.png"

def load_font(size, bold=False):
    regular_candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/System/Library/Fonts/Helvetica.ttc" # Mac fallback
    ]
    bold_candidates = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "/System/Library/Fonts/Helvetica-Bold.ttc" # Mac fallback
    ]

    for path in (bold_candidates if bold else regular_candidates):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)

    return ImageFont.load_default()

def wrap_text(draw_obj, text, font_obj, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        trial = f"{current} {word}".strip()
        left, _, right, _ = draw_obj.textbbox((0, 0), trial, font=font_obj)
        if right - left <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines

def format_class_name(name):
    # Cleans up the string formatting nicely
    return str(name).replace("___", " • ").replace("__", " ").replace("_", " ")

# Setup Canvas
width, height = 1600, 1050
canvas = Image.new("RGB", (width, height), (248, 250, 252))
draw = ImageDraw.Draw(canvas)

# Load Fonts
title_font = load_font(56, bold=True)
section_font = load_font(30, bold=True)
main_font = load_font(26)
small_font = load_font(22)
bold_small_font = load_font(22, bold=True)

# Main background card wrapper
card_margin = 36
draw.rounded_rectangle(
    [card_margin, card_margin, width - card_margin, height - card_margin],
    radius=30,
    fill=(255, 255, 255),
    outline=(226, 232, 240),
    width=3
)

draw.text((80, 60), "Plant Disease Prediction Report", fill=(15, 23, 42), font=title_font)

# Confidence color logic
confidence_pct = round(confidence.item() * 100, 2)
confidence_color = (16, 185, 129) if confidence_pct >= 80 else (217, 119, 6) if confidence_pct >= 60 else (220, 38, 38)

# 1. Primary Result Card
draw.rounded_rectangle(
    [80, 140, 1520, 280],
    radius=24,
    fill=(248, 250, 252),
    outline=(203, 213, 225),
    width=2
)

# Clean up predicted class name
display_pred_class = format_class_name(predicted_class)

draw.text((110, 170), "Predicted Class", fill=(71, 85, 105), font=small_font)
draw.text((110, 205), display_pred_class, fill=(15, 23, 42), font=section_font)

# Dynamically place the "TOP MATCH" badge next to the text
pred_bbox = draw.textbbox((0, 0), display_pred_class, font=section_font)
pred_width = pred_bbox[2] - pred_bbox[0]
badge_x = 110 + pred_width + 24
badge_y = 205

draw.rounded_rectangle(
    [badge_x, badge_y, badge_x + 140, badge_y + 38],
    radius=19,
    fill=(220, 252, 231)
)
draw.text(
    (badge_x + 16, badge_y + 6),
    "TOP MATCH",
    fill=(22, 163, 74),
    font=load_font(18, bold=True)
)

# Dynamically align Confidence to the right
conf_str = f"{confidence_pct:.2f}%"
conf_val_font = load_font(46, bold=True)
conf_bbox = draw.textbbox((0, 0), conf_str, font=conf_val_font)
conf_width = conf_bbox[2] - conf_bbox[0]

# Right margin is 1520 - 40px padding
right_align_x = 1480 
conf_x = right_align_x - conf_width

# Align indicator dot to the left of the percentage
dot_radius = 10
dot_x = conf_x - 30
dot_y = 212 + (46 // 2) - dot_radius 

draw.ellipse([dot_x, dot_y, dot_x + dot_radius*2, dot_y + dot_radius*2], fill=confidence_color)
draw.text((conf_x, 200), conf_str, fill=confidence_color, font=conf_val_font)

lbl_bbox = draw.textbbox((0, 0), "Confidence", font=small_font)
lbl_width = lbl_bbox[2] - lbl_bbox[0]
lbl_x = right_align_x - lbl_width
draw.text((lbl_x, 170), "Confidence", fill=(71, 85, 105), font=small_font)

# Dynamic Y-spacing based on disease info availability
current_y = 320

# 2. Disease Information Card (Optional)
if predicted_class in disease_info:
    draw.rounded_rectangle(
        [80, current_y, 1520, current_y + 200],
        radius=20,
        fill=(249, 250, 251),
        outline=(229, 231, 235),
        width=2
    )
    draw.text((110, current_y + 30), "Disease Information", fill=(55, 65, 81), font=bold_small_font)
    wrapped_info = wrap_text(draw, disease_info[predicted_class], main_font, max_width=1300)
    
    y_info = current_y + 70
    for line in wrapped_info[:4]: # Limit to 4 lines to fit safely
        draw.text((110, y_info), line, fill=(31, 41, 55), font=main_font)
        y_info += 34
    
    current_y += 240 # Push the bar chart further down

# 3. Top-5 Probabilities Bar Chart
draw.text((80, current_y), f"Top-{top_k} Class Probabilities", fill=(30, 41, 59), font=section_font)

# Chart layout variables
chart_left = 650  # Increased to prevent text clipping
chart_right = 1420
bar_top_start = current_y + 60
bar_height = 42   # Slightly slimmer bars
bar_radius = 21
gap = 26

for i in range(top_k):
    cls_name = str(class_names[top_indices[i]])
    prob_pct = float(top_probs[i] * 100)

    y_top = bar_top_start + i * (bar_height + gap)
    y_bottom = y_top + bar_height

    display_name = format_class_name(cls_name)

    # Truncate text nicely
    max_chars = 42
    if len(display_name) > max_chars:
        display_name = display_name[:max_chars] + "..."

    # Class Name Label
    draw.text(
        (80, y_top + 8),
        f"{i+1}. {display_name}",
        fill=(15, 23, 42),
        font=small_font
    )

    # Bar Background
    draw.rounded_rectangle(
        [chart_left, y_top, chart_right, y_bottom],
        radius=bar_radius,
        fill=(241, 245, 249) # Taildwind Slate-100
    )

    # Bar Fill 
    fill_width = int((chart_right - chart_left) * (prob_pct / 100.0))
    if fill_width > 0:
        # Prevent PIL from warping tiny rounded rectangles
        if fill_width < (bar_radius * 2):
            fill_width = bar_radius * 2 
            
        draw.rounded_rectangle(
            [chart_left, y_top, chart_left + fill_width, y_bottom],
            radius=bar_radius,
            fill=(59, 130, 246) # Tailwind Blue-500
        )

    # Percentage Text
    draw.text(
        (1445, y_top + 8),
        f"{prob_pct:.2f}%",
        fill=(71, 85, 105),
        font=bold_small_font
    )

# Footer Divider
draw.line(
    [70, 950, 1510, 950],
    fill=(226, 232, 240),
    width=2
)

footer_text = "Powered by PyTorch • Prediction Info"
bbox = draw.textbbox((0, 0), footer_text, font=small_font)
text_width = bbox[2] - bbox[0]
footer_x = (width - text_width) // 2

draw.text(
    (footer_x, 970),
    footer_text,
    fill=(148, 163, 184),
    font=small_font
)

canvas.save(chart_path)
print(f"\nPrediction chart saved at: {chart_path}")