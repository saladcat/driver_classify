import matplotlib.pyplot as plt
import torch
from train import *
import cv2
from torchvision import transforms
from PIL import Image
import torchvision

device = torch.device("cpu")


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def test_model(model, img):
    was_training = model.training
    model.eval()
    with torch.no_grad():
        input = data_transforms['val'](img)
        shape = input.shape
        input = input.resize_(1, *shape)
        input = input.to(device)
        output = model(input)
        _, preds = torch.max(output, 1)

    model.train(mode=was_training)
    return class_names[preds[0]]


if __name__ == '__main__':
    model = torchvision.models.squeezenet1_1(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 3)

    final_conv = nn.Conv2d(512, 3, kernel_size=1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        final_conv,
        nn.ReLU(inplace=True),
        nn.AvgPool2d(13, stride=1)
    )
    model.num_classes = 3

    model = model.to(device)

    model.load_state_dict(torch.load('model/params.pkl'))
    now = time.time()
    print("dsadsa")
    for i in range(30):
        img = Image.open("b.jpg")
        a = test_model(model, img)
        # print('test:{}'.format(test_model(model, img)))
    print("test_time={}".format((time.time() - now) / 30))
