import math

def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    """
    Implements the cosine annealing learning rate schedule with warm-up.

    This function calculates the learning rate at a given iteration 't' based on
    three distinct phases: warm-up, cosine annealing, and post-annealing.

    Args:
        t (int): The current iteration (step) number.
        alpha_max (float): The maximum (initial) learning rate.
        alpha_min (float): The minimum (final) learning rate after annealing.
        T_w (int): The number of warm-up iterations.
        T_c (int): The total number of cosine annealing iterations (including warm-up).

    Returns:
        float: The calculated learning rate at iteration 't'.
    """
    # Ensure that T_w is less than or equal to T_c for logical schedule progression.
    # If T_w is greater than T_c, it implies an invalid schedule setup.
    if T_w > T_c:
        print("Warning: Warm-up iterations (T_w) should not be greater than total cosine iterations (T_c).")
        # In this case, we might default to alpha_min or handle as an error,
        # but for this assignment, we'll proceed with the defined phases as much as possible.

    alpha_t = 0.0  # Initialize learning rate

    # --- Warm-up Phase ---
    # If the current iteration 't' is less than the warm-up period 'T_w',
    # the learning rate increases linearly from 0 to alpha_max.
    if t < T_w:
        # Avoid division by zero if T_w is 0. If T_w is 0, warm-up is skipped,
        # and we immediately go into cosine annealing or post-annealing.
        if T_w == 0:
            alpha_t = alpha_max
        else:
            alpha_t = (t / T_w) * alpha_max
    # --- Cosine Annealing Phase ---
    # If the current iteration 't' is within or at the start of the annealing period,
    # and also greater than or equal to the warm-up period 'T_w'.
    # The learning rate decays from alpha_max down to alpha_min following a cosine curve.
    elif T_w <= t <= T_c:
        # Calculate the progress within the annealing phase.
        # This scales the current step 't' from the end of warm-up to the end of annealing
        # into a [0, 1] range, which is then mapped by the cosine function.
        progress = (t - T_w) / (T_c - T_w) if (T_c - T_w) > 0 else 0
        # The cosine annealing formula:
        # alpha_min + 0.5 * (1 + cos(progress * pi)) * (alpha_max - alpha_min)
        # This formula smoothly transitions the learning rate from alpha_max (when progress is 0)
        # to alpha_min (when progress is 1).
        alpha_t = alpha_min + 0.5 * (1 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
    # --- Post-annealing Phase ---
    # If the current iteration 't' is greater than the total cosine annealing iterations 'T_c',
    # the learning rate is fixed at the minimum value 'alpha_min'.
    else:  # t > T_c
        alpha_t = alpha_min

    return alpha_t