"""Pure Plan5 algorithm operators used by numerical unit tests."""
import torch


def ddqn_target(reward, online_next_q, target_next_q, gamma=0.95,
                terminated=False, truncated=False):
    """Return Double-DQN targets; truncation bootstraps, termination does not."""
    online_next_q = torch.as_tensor(online_next_q)
    target_next_q = torch.as_tensor(
        target_next_q, dtype=online_next_q.dtype,
        device=online_next_q.device,
    )
    reward = torch.as_tensor(
        reward, dtype=target_next_q.dtype, device=target_next_q.device,
    )
    action = torch.argmax(online_next_q, dim=-1)
    value = target_next_q.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    terminated = torch.as_tensor(
        terminated, dtype=torch.bool, device=value.device,
    )
    # ``truncated`` is deliberately accepted but not masked: SUMO's fixed
    # horizon is a time-limit truncation and therefore uses V/Q bootstrap.
    torch.as_tensor(truncated, dtype=torch.bool, device=value.device)
    return reward + float(gamma) * (~terminated).to(value.dtype) * value


def dqn_full_vector_target(predicted_q, actions, scalar_targets):
    """Preserve the original DQN full-vector mean-MSE target semantics."""
    predicted_q = torch.as_tensor(predicted_q)
    actions = torch.as_tensor(
        actions, dtype=torch.long, device=predicted_q.device,
    ).reshape(-1, 1)
    scalar_targets = torch.as_tensor(
        scalar_targets, dtype=predicted_q.dtype, device=predicted_q.device,
    ).reshape(-1, 1)
    if predicted_q.ndim != 2 or actions.shape[0] != predicted_q.shape[0] \
            or scalar_targets.shape[0] != predicted_q.shape[0]:
        raise ValueError('DQN full-vector target batch dimensions mismatch')
    target = predicted_q.detach().clone()
    target.scatter_(1, actions, scalar_targets)
    return target
