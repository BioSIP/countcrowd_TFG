
from torch.utils.tensorboard import SummaryWriter

expcode = f'{opciones}_model'
writer = SummaryWriter(os.path.join(wdir, 'runs', expcode))

model = Model()
images, labels = next(iter(dataloader))
writer.add_graph(model,images)

for epoch in range(20):
    total_epoch_loss_train = 0.0
    total_epoch_loss_val = 0.0
    # minibatch loop: 
        # initializae loss and data x, y 
        # forwadr pass yhat = model(x)
        # compute loss
        # backward pass
        # update loss: total_epoch_loss_train +=loss.. 
    writer.add_scalar('Loss/train', total_epoch_loss_train, epoch)

    # validation:
    writer.add_scalar('Loss/val', total_epoch_loss_val, epoch)
writer.close()