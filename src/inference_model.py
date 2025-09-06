from ultralytics import YOLO

def run_inference(image_path='imagem.jpg'):
    model = YOLO('best.pt') # passível de mudança, não sei o local de armazenamento do modelo
    results = model(image_path)

    results.show()
    results.save(filename='output.jpg')

if __name__ == '__main__': # não é necessário, mas é uma boa prática
    run_inference()