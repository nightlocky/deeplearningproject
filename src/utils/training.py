def train_autoencoder(model, train_loader, optimizer, criterion, device, epochs, tracker=None):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        batch_losses = []
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)

        for imgs, _ in loop:
            imgs = imgs.to(device)

            recon = model(imgs)
            loss = criterion(recon, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            loop.set_postfix(loss=loss.item())

        avg_loss = np.mean(batch_losses)
        train_losses.append(avg_loss)

        if tracker:
            tracker.log_metric("train_loss", avg_loss, step=epoch)

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    return train_losses