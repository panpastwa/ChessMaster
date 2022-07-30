import torch
from torch.nn.functional import cosine_similarity
from matplotlib import pyplot as plt

from src.utils.general import fens_extract_embeddings


def cosine_similarity_between_moves(model, fens, device, move=20):

    cos_sims = torch.tensor([], device=device, dtype=torch.double)
    for game in fens:
        # Ignore too short games
        if len(game) < 2*move+1:
            continue
        emb = fens_extract_embeddings(model, game[:2*move+1], device)
        cos_sims = torch.cat((cos_sims, cosine_similarity(emb, emb[move]).unsqueeze(dim=0)), dim=0)

    mean, std = cos_sims.mean(dim=0).cpu(), cos_sims.std(dim=0).cpu()
    return mean, std


def cosine_similarity_between_games(model, model_game_fens, fens, device, max_move=20):

    model_game_emb = fens_extract_embeddings(model, model_game_fens[:max_move], device)
    cos_sims = torch.tensor([], device=device, dtype=torch.double)

    for game in fens:
        # Ignore too short games
        if len(game) < max_move:
            continue
        emb = fens_extract_embeddings(model, game[:max_move], device)
        cos_sims = torch.cat((cos_sims, cosine_similarity(model_game_emb, emb).unsqueeze(dim=0)), dim=0)

    return cos_sims
    # mean, std = cos_sims.mean(dim=0).cpu(), cos_sims.std(dim=0).cpu()
    # return mean, std


def plot_cosine_similarity_between_moves(mean, std, fill_between=False, show=False, save=None):
    plt.plot(range(mean.shape[0]), mean)
    plt.ylabel("Cosine similarity")
    plt.xlabel("Move number")
    if fill_between:
        plt.fill_between(range(mean.shape[0]), mean - std, mean + std, alpha=0.2)
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()


def plot_cosine_similarity_between_games(cos_sims, labels=None, fill_between=False, show=False, save=None):
    plt.figure(figsize=(12, 9))
    for i, cos_sim in enumerate(cos_sims):
        plt.plot(range(cos_sim.shape[0]), cos_sim)
    plt.ylabel("Cosine similarity")
    plt.xlabel("Move number")
    if labels is not None:
        plt.legend(labels)
    # if fill_between:
    #     plt.fill_between(range(mean.shape[0]), mean - std, mean + std, alpha=0.2)
    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()
