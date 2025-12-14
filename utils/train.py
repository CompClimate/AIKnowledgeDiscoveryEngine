# trains models
# returns training loss and val loss

def train():
    # Training loop
    losses = []
    val_losses = []
    n_epochs = config.getint('TRAINING', 'epochs')
    loss_fn = config['TRAINING']['loss_fn']
    loss_fn = getattr(torch.nn, loss_fn)()

    start_epoch = 0

    for epoch in range(start_epoch, n_epochs):
        model.train(True)
        loss_final = 0
        train_loss_accum = None
        for batch in train_loader:
            batch = feature_standardizer(batch)
            if config['TRAINING']['machine'] == 'gpu':
                batch.to(DEVICE)
            pred = model(batch)
            loss = loss_fn(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            n_snaps = 0
            for val_batch in val_loader:
                val_batch = feature_standardizer(val_batch)
                if config['TRAINING']['machine'] == 'gpu':
                    val_batch.to(DEVICE)
                pred = model(val_batch)
                val_loss += loss_fn(pred, clustered_target.x)
                n_snaps += 1
            val_loss /= n_snaps
            scheduler.step()
        output = config['OUTPUT']['dir']
        if epoch % 5 == 0:
            print(f"epoch: {epoch}; loss: {loss_mean:.5f}; val_loss: {val_loss_mean:.5f}")
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
        losses.append(loss_mean.item())
        val_losses.append(val_loss_mean.item())
        if epoch % config.getint('OUTPUT', 'n_epochs_between_checkpoints') == 0:
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{output}/{save_model}_epoch{epoch}.pt")
            torch.save(losses, f"{output}/losses.pt")
            torch.save(val_losses, f"{output}/val_losses.pt")
