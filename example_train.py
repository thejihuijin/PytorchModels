from DeepCodec import *

def main():
    # Set up model
    N = 512
    r = 8
    net = DeepCodec(r)
    GPU = False
    print(GPU)
    if GPU:
        net.cuda()


    # Prepare Data
    #transform = transforms.Compose([transforms.ToTensor(),
    #                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    transform = transforms.ToTensor()
    trainset = torchvision.datasets.CIFAR10(root='./data',train=True, 
                                        download = True, transform=transform)

    DC_data = torch.zeros(2*len(trainset),512,1)
    for i,data in enumerate(trainset):
        images, labels = data
        gray = rgbtogray(images)
        gray = (gray-.5)*2
        H,W = gray.size()
        H2 = int(H/2)
        DC_data[2*i] = gray[:H2,:].view(512,1)
        DC_data[2*i+1] = gray[H2:,:].view(512,1)

    DC_trainset = torch.utils.data.TensorDataset(DC_data,DC_data)
    DC_trainloader = torch.utils.data.DataLoader(DC_trainset, batch_size=1000,shuffle=True)


    # Train Model
    train_net(net, DC_trainloader, num_epochs=5000,GPU=GPU, 
              weightpath='./weights/',save_epoch=250,
              lr=0.001,momentum=0.9)

if __name__ == '__main__':
    main()
