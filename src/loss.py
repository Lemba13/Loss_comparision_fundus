import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dice_loss(input: torch.Tensor, target: torch.Tensor, smooth: float = 1) -> torch.Tensor:
    """
    Compute Dice Loss.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.
        smooth (float, optional): Smoothing factor. Defaults to 1.

    Returns:
        torch.Tensor: Dice Loss.
    """
    smooth = 1
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    dice = (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)

    return 1 - dice


def bce_and_dice(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute BCE and Dice Loss combined.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Combined loss.
    """
    bce_loss = F.binary_cross_entropy(input=input, target=target)
    d_loss = dice_loss(input=input, target=target)

    return 0.5 * bce_loss + d_loss


def aceloss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute ACE Loss.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: ACE Loss.
    """
    x = target[:, :, 1:, :] - target[:, :, :-1, :]
    y = target[:, :, :, 1:] - target[:, :, :, :-1]

    delta_x = x[:, :, 1:, :-2] ** 2
    delta_y = y[:, :, :-2, 1:] ** 2

    delta_u = torch.abs(delta_x + delta_y)

    length = torch.mean(torch.sqrt(delta_u + 0.00000001))

    C_1 = torch.ones_like(target).to(device)
    C_2 = torch.zeros_like(target).to(device)

    region_in = torch.abs(
        torch.mean(target.to(device) * ((input.to(device) - C_1) ** 2))
    )

    region_out = torch.abs(
        torch.mean((1 - target.to(device)) * ((input.to(device) - C_2) ** 2))
    )

    lambdaP = 10
    mu = 1

    sc = length + lambdaP * (mu * region_in + region_out)

    return sc


def bce_ace_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute BCE and ACE Loss combined.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Combined loss.
    """
    ace = aceloss(input=input, target=target)
    bce = F.binary_cross_entropy(input=input, target=target)

    return ace + bce


def dice_coeff(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Dice Coefficient.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Dice Coefficient.
    """
    smooth = 1
    iflat = input.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)


def bce(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Binary Cross Entropy (BCE) Loss.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: BCE Loss.
    """
    return F.binary_cross_entropy(input=input, target=target)


def focal_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Focal Loss.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Focal Loss.
    """
    alpha = 0.8
    gamma = 2
    input = input.view(-1)
    target = target.view(-1)
    target = (target > (19 / 255)).float()

    BCE = F.binary_cross_entropy(input, target, reduction="mean")
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

    return focal_loss


def tversky_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Tversky Loss.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Tversky Loss.
    """
    smooth = 1
    alpha = 0.5
    beta = 0.5

    inputs = input.view(-1)
    targets = target.view(-1)

    TP = (inputs * targets).sum()
    FP = ((1 - targets) * inputs).sum()
    FN = (targets * (1 - inputs)).sum()

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - Tversky


def weighted_bce(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Weighted BCE Loss.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Weighted BCE Loss.
    """
    target = (target > (19 / 255)).float()
    beta = torch.tensor(0.7).cuda()
    loss = F.binary_cross_entropy(input, target, weight=beta)
    return loss


def balanced_bce(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute Balanced BCE Loss.

    Args:
        input (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Target tensor.

    Returns:
        torch.Tensor: Balanced BCE Loss.
    """
    input = input.view(-1)
    target = target.view(-1)
    target = (target > (10 / 255)).float()
    beta = 1 - (target.sum() / int(list(target.size())[0]))

    loss = F.binary_cross_entropy(input, target, weight=beta)
    return loss
