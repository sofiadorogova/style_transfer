import torch
import torch.nn as nn

"""
Dicriminator
"""
class VGGBlock(nn.Module):
    """
    Один блок:
    - Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU
    - AvgPool2d(kernel_size=2, stride=2)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        return self.block(x)

class VGGFeatureExtractor(nn.Module):
    """
    Последовательно применяет 5 блоков, начиная с 3 каналов (RGB) и 
    каждый раз удваивая число каналов:
      3 -> 6 -> 12 -> 24 -> 48 -> 96.
    После 5 блоков при входном размере 512 * 512 получим выход:
      (batch_size = 1, 96, 16, 16).
    """
    def __init__(self):
        super().__init__()
        # Блоки: (3->6), (6->12), (12->24), (24->48), (48->96)
        self.block0 = VGGBlock(in_channels=3,  out_channels=6)
        self.block1 = VGGBlock(in_channels=6,  out_channels=12)
        self.block2 = VGGBlock(in_channels=12, out_channels=24)
        self.block3 = VGGBlock(in_channels=24, out_channels=48)
        self.block4 = VGGBlock(in_channels=48, out_channels=96)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

class VGGClassifier(nn.Module):
    """
    Разворачивает тензор и пропускает через 2 полносвязных слоя:
    Linear(96 * 16 * 16 -> 4096) -> LeakyReLU -> Linear(4096 -> 1) -> Sigmoid
    (число 96 * 16 * 16 для входа 512*512)
    """
    def __init__(self, in_features=96*16*16):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 96, 16, 16) -> (B, 96*16*16)
            nn.Linear(in_features, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

class VGGDiscriminator(nn.Module):
    """
    Итоговая модель дискриминатор:
    - Выделяем признаки с помощью VGGFeatureExtractor
    - Классифицируем через VGGClassifier
    """
    def __init__(self):
        super().__init__()
        self.features = VGGFeatureExtractor()
        self.classifier = VGGClassifier(in_features=96*16*16)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

"""
Generator
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, hid_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()

        self.conv_block = nn.Sequential(*[nn.Conv2d(in_channels=in_channels, 
                                                    out_channels=hid_channels, 
                                                    kernel_size=kernel_size, 
                                                    padding=padding),
                                          nn.BatchNorm2d(hid_channels),
                                          nn.ReLU(),
                                          nn.Conv2d(in_channels=hid_channels, 
                                                    out_channels=out_channels, 
                                                    kernel_size=kernel_size, 
                                                    padding=padding),
                                          nn.BatchNorm2d(out_channels),
                                          nn.ReLU()])
        
    def forward(self, x):
        return self.conv_block(x)
    
class Upsampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(*[nn.Upsample(scale_factor=2),
                                        nn.Conv2d(in_channels=in_channels, 
                                                  out_channels=out_channels, 
                                                  kernel_size=2, 
                                                  padding='same')])
        
    def forward(self, x):
        return self.upsample(x)
    
class UNet_Encoder(nn.Module):
    def __init__(self, init_channels):
        super().__init__()

        self.conv0 = ConvBlock(in_channels=init_channels, hid_channels=64, out_channels=64)
        self.conv1 = ConvBlock(in_channels=64, hid_channels=128, out_channels=128)
        self.conv2 = ConvBlock(in_channels=128, hid_channels=256, out_channels=256)
        self.conv3 = ConvBlock(in_channels=256, hid_channels=512, out_channels=512)
        self.conv4 = ConvBlock(in_channels=512, hid_channels=1024, out_channels=1024)
        
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        e0 = self.conv0(x)
        e1 = self.conv1(self.pooling(e0))
        e2 = self.conv2(self.pooling(e1))
        e3 = self.conv3(self.pooling(e2))
        e4 = self.conv4(self.pooling(e3))

        encoder_outputs = [e0, e1, e2, e3]
        return e4, encoder_outputs
    
class UNet_Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.up0 = Upsampling(in_channels=1024, out_channels=512)
        self.up1 = Upsampling(in_channels=512, out_channels=256)
        self.up2 = Upsampling(in_channels=256, out_channels=128)
        self.up3 = Upsampling(in_channels=128, out_channels=64)

        self.deconv0 = ConvBlock(in_channels=1024, hid_channels=512, out_channels=512)
        self.deconv1 = ConvBlock(in_channels=512, hid_channels=256, out_channels=256)
        self.deconv2 = ConvBlock(in_channels=256, hid_channels=128, out_channels=128)
        self.deconv3 = ConvBlock(in_channels=128, hid_channels=64, out_channels=64)

        self.final = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x, encoder_outputs):
        d0 = self.up0(x)
        d0 = torch.cat([encoder_outputs[3], d0], dim = 1)
        d0 = self.deconv0(d0)

        d1 = self.up1(d0)
        d1 = torch.cat([encoder_outputs[2], d1], dim = 1)
        d1 = self.deconv1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([encoder_outputs[1], d2], dim = 1)
        d2 = self.deconv2(d2)

        d3 = self.up3(d2)
        d3 = torch.cat([encoder_outputs[0], d3], dim = 1)
        d3 = self.deconv3(d3)

        return self.final(d3)
    
class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()

        self.encoder = UNet_Encoder(init_channels=n_channels)
        self.decoder = UNet_Decoder(num_classes=n_classes)
        
    def forward(self, x):
        x, encoder_outputs = self.encoder(x)
        return self.decoder(x, encoder_outputs)




if __name__ == "__main__":
    generator = UNet(n_channels=3, n_classes=3)
    discriminator = VGGDiscriminator()

    test_input = torch.randn(1, 3, 512, 512)
  
    fake_image = generator(test_input)
    print("Generator output shape:", fake_image.shape)

    disc_output = discriminator(fake_image)
    print("Discriminator output shape:", disc_output.shape)
    print("Discriminator output:", disc_output)