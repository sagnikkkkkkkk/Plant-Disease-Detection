from PIL import Image

frames = []
paths = [
    'assets/accuracy.png',
    'assets/loss.png',
    'assets/confusion_matrix.png',
    'assets/prediction_dashboard.png'
]
for p in paths:
    try:
        im = Image.open(p).convert('RGBA').resize((800,600))
        frames.append(im)
    except Exception as e:
        print('skip', p, e)

if not frames:
    print('No frames found to create GIF')
else:
    frames[0].save('assets/demo.gif', format='GIF', append_images=frames[1:], save_all=True, duration=1200, loop=0)
    print('assets/demo.gif created')
