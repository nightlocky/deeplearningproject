def get_reconstruction_errors(model, loader, device):
    model.eval()
    errors, labels = [], []

    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc="Extracting errors"):
            imgs = imgs.to(device)
            recon = model(imgs)

            batch_err = torch.mean((imgs - recon)**2, dim=[1,2,3]).cpu().numpy()

            errors.extend(batch_err)
            labels.extend(lbls.numpy())

    return np.array(errors), np.array(labels)