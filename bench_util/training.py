import time
import torch
import torch.cuda.amp as amp

try:
    from tqdm import trange
except ImportError:
    trange = range


def train_epoch_mixed(model, loader, optimizer, loss_function, scaler, device):

    # set the model to train mode
    model.train()

    iteration_times = []
    # iterate over the batches of this epoch
    for x, y in loader:
        t0 = time.time()
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        with amp.autocast():
            prediction = model(x)
            loss = loss_function(prediction, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        iteration_times.append(time.time() - t0)

    return iteration_times


def train_epoch_default(model, loader, optimizer, loss_function, scaler, device):

    # set the model to train mode
    model.train()

    iteration_times = []
    # iterate over the batches of this epoch
    for x, y in loader:
        t0 = time.time()
        # move input and target to the active device (either cpu or gpu)
        x, y = x.to(device), y.to(device)

        # zero the gradients for this iteration
        optimizer.zero_grad()

        prediction = model(x)
        loss = loss_function(prediction, y)

        loss.backward()
        optimizer.step()
        iteration_times.append(time.time() - t0)

    return iteration_times


def train_loop(model, loader, loss_function, n_epochs,
               precision='mixed'):

    device = torch.device('cuda')
    model = model.to(device)
    loss_function = loss_function.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if precision == 'mixed':
        scaler = amp.GradScaler()
        train_epoch = train_epoch_mixed
    else:
        scaler = None
        train_epoch = train_epoch_default

    t_tot = time.time()
    iteration_times = []
    for epoch in trange(n_epochs):
        this_runtimes = train_epoch(model, loader, optimizer,
                                    loss_function, scaler, device)
        iteration_times.extend(this_runtimes)
    t_tot = time.time() - t_tot

    return t_tot, iteration_times
