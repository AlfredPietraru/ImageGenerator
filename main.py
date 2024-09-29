import torch
from net import Generator, Discriminator
import preprocessing as prep
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms.functional import rotate

EPOCHS = 2
LEARNING_RATE = 0.00005
BATCH_SIZE = 40
TRANFORMATIONS = 4
generator = Generator()
discriminator = Discriminator()
gen_optimizer = torch.optim.Adam(params=generator.parameters(), lr=LEARNING_RATE)
disc_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=LEARNING_RATE)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, generator : Generator):
        self.root_dir = prep.INITIAL_IMAGES_PATH
        self.generator = generator

    def __len__(self):
        elements  = os.listdir(self.root_dir)
        return len(elements)
    
  
    def __getitem__(self, idx):
        images = (prep.get_one_training_image(idx) - 128) / 128
        images = torch.stack([images, rotate(images, 90), rotate(images, 180), rotate(images, 270)], dim=0)
        generated_image  : torch.Tensor = generator(torch.normal(0, 0.1, size=(TRANFORMATIONS, 100,)))
        return  images, generated_image.squeeze(dim=0)


def discriminator_loss_function(real, generated):
    return torch.sum(-torch.log(1 - generated) - torch.log(real))

def generator_loss_function(generated):
    return torch.sum(torch.log(1 - generated))


def training_loop(epochs : int, batch_numbers : int):
    if (os.path.isfile("./generator.pth")):
        generator.load_state_dict(torch.load("./generator.pth", weights_only=True))
    if (os.path.isfile("./discriminator.pth")):
        discriminator.load_state_dict(torch.load("./discriminator.pth", weights_only=True))
    generator.train()
    discriminator.train()
    dataset = ImageDataset(generator)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_result = torch.zeros(size=(epochs * batch_numbers * BATCH_SIZE,))
    for j in range(epochs):
        for idx, data in enumerate(dataloader):
            images, generated_images = data
            images = images.reshape(BATCH_SIZE * TRANFORMATIONS, images.shape[2], images.shape[3], images.shape[4])
            generated_images = generated_images.reshape(BATCH_SIZE * TRANFORMATIONS, generated_images.shape[2],
                                                         generated_images.shape[3], generated_images.shape[4])
            gen_optimizer.zero_grad()
            generator_loss = generator_loss_function(discriminator(generated_images))
            generator_loss.backward()
            gen_optimizer.step()

            disc_optimizer.zero_grad()
            discriminator_loss = discriminator_loss_function(discriminator(images),
                                                          discriminator(generated_images.detach()))
            print(generator_loss, discriminator_loss)
            discriminator_loss.backward()
            disc_optimizer.step()
            train_result[j * batch_numbers * BATCH_SIZE + idx] = generator_loss
        torch.save(generator.state_dict(), "./generator.pth")
        torch.save(discriminator.state_dict(), "./discriminator.pth")
        print("s-a salvat")
    plt.plot(range(0, EPOCHS * batch_numbers * BATCH_SIZE), train_result.detach().numpy())
    plt.show()

def eval_way():
    generator.load_state_dict(torch.load("./generator.pth"))
    discriminator.load_state_dict(torch.load("./discriminator.pth"))
    discriminator.eval()
    generator.eval()
    x = torch.normal(0, 0.1, size=(1, 100))
    out = generator(x) * 128 + 128
    prep.show_image(out.squeeze(dim=0))


training_loop(EPOCHS, 10)
# eval_way()

