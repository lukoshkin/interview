import copy
import torch

def _name(instance):
    try: return instance.__name__
    except: return instance.__class__.__name__


def train_open_eyes_clf(
    net, train_criterion, optimizer, train_bl, val_bl,
    scheduler=None, device='cpu', val_criterion=None, epochs=10,
    continue_val_score=float('inf'), verbose=True):

    if verbose:
        headline = 'Epoch\t\tTrain\t\tValidation\n'
        if val_criterion is not None:
            headline += '\t\t\t\t'
            headline += f'{_name(train_criterion)}\t'
            headline += f'{_name(val_criterion)}\n\n'

        print(headline)

    best_val_score = continue_val_score
    best_model_state = None

    for epoch in range(1, epochs+1):
        train_loss = val_loss = 0
        net.train()

        for X,y in train_bl:
            optimizer.zero_grad()

            X, y = X.to(device), y.to(device)

            loss = train_criterion(net(X), y)
            loss.backward()
            train_loss += loss.item() * len(y)/train_bl.batch_size

            optimizer.step()
        train_loss /= len(train_bl)

        net.eval()
        Y = torch.Tensor([])
        Y_pred = torch.Tensor([])

        for X,y in val_bl:
            Y = torch.cat((Y, y))
            X, y = X.to(device), y.to(device)

            logits = net(X)
            loss = train_criterion(logits, y)
            
            y_pred = logits.detach().sigmoid()
            Y_pred = torch.cat((Y_pred, y_pred.cpu()))

            val_loss += loss.item() * len(y)/val_bl.batch_size

        val_loss /= len(val_bl)
        val_score = val_loss
        if val_criterion is not None:
            val_score = val_criterion(Y, Y_pred)

        if val_score < best_val_score:
            best_val_score = val_score
            best_model_state = copy.deepcopy(net.state_dict())

        if scheduler is not None: scheduler.step()

        freq = max(epochs//20, 1)
        if verbose and epoch % freq == 0:
            logline = f'{epoch}/{epochs}\t\t{train_loss:.4}\t\t{val_loss:.4f}'
            if val_criterion is not None: logline += f'\t{val_score:.4}'
            print(logline)

    return best_val_score, best_model_state
