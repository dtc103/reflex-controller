import torch
torch.set_default_dtype(torch.float32)

class MuscleModel:
    def __init__(self, muscle_params, num_joints, nenvironments):
        self.device = "cuda"
        self.nactioncount = action_tensor.shape[1]
        self.nenvironments = nenvironments

        for k, v in muscle_params.items():
            setattr(self, k, v)

        self.phi_min = (
            torch.ones(
                (nenvironments, action_tensor.shape[1] // 2),
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            * self.phi_min
        )
        self.phi_max = (
            torch.ones(
                (nenvironments, action_tensor.shape[1]// 2),
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            * self.phi_max
        )
        self.activation_tensor = torch.zeros_like(
            action_tensor,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_tensor = torch.zeros_like(
            action_tensor,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_dot_tensor = torch.zeros_like(
            action_tensor,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.force_tensor = torch.zeros_like(
            action_tensor,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_1_tensor = torch.zeros(
            (self.nenvironments, action_tensor.shape[1]// 2),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.lce_2_tensor = torch.zeros(
            (self.nenvironments, action_tensor.shape[1]// 2),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )
        self.moment, self.lce_ref = self.compute_parametrization(action_tensor)


    def FL(self, lce: torch.Tensor) -> torch.Tensor:
        """
        Force length
        """
        length = lce
        b1 = self.bump(length, self.lmin, 1, self.lmax)
        b2 = self.bump(length, self.lmin, 0.5 * (self.lmin + 0.95), 0.95)
        bump_res = b1 + 0.15 * b2
        return bump_res

    def bump(self, length: torch.Tensor, A: float, mid: float, B: float) -> torch.Tensor:
        """
        skewed bump function: quadratic spline
        Input:
            :length: tensor of muscle lengths [Nenv, Nactuator]
            :A: scalar
            :mid: scalar
            :B: scalar

        Returns:
            :torch.Tensor: contains FV result [Nenv, Nactuator]
        """

        left = 0.5 * (A + mid)
        right = 0.5 * (mid + B)
        # Order of assignment needs to be inverse to the if-else-clause case
        bump_result = torch.ones_like(
            length,
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        x = (B - length) / (B - right)
        bump_result = 0.5 * x * x

        x = (length - mid) / (right - mid)
        bump_result = torch.where(length < right, 1 - 0.5 * x * x, bump_result)

        x = (mid - length) / (mid - left)
        bump_result = torch.where(length < mid, 1 - 0.5 * x * x, bump_result)

        x = (length - A) / (left - A)
        bump_result = torch.where((length < left) & (length > A), 0.5 * x * x, bump_result)

        bump_result = torch.where(
            torch.logical_or((length <= A), (length >= B)),
            torch.tensor([0], dtype=torch.float32, device=self.device),
            bump_result,
        )
        return bump_result

    def compute_parametrization(self, action_tensor):
        """
        Find parameters for muscle length computation.
        This should really only be done once...

        We compute them as one long vector now.
        """
        moment = torch.zeros_like(action_tensor)
        moment[:, : int(moment.shape[1] // 2)] = (self.lce_max - self.lce_min + self.eps) / (self.phi_max - self.phi_min + self.eps)
        moment[:, int(moment.shape[1] // 2) :] = (self.lce_max - self.lce_min + self.eps) / (self.phi_min - self.phi_max + self.eps)
        lce_ref = torch.zeros_like(action_tensor)
        lce_ref[:, : int(lce_ref.shape[1] // 2)] = self.lce_min - moment[:, : int(moment.shape[1] // 2)] * self.phi_min
        lce_ref[:, int(lce_ref.shape[1] // 2) :] = self.lce_min - moment[:, int(moment.shape[1] // 2) :] * self.phi_max
        return moment, lce_ref

    def compute_virtual_lengths(self, actuator_pos: torch.Tensor) -> None:
        """
        Compute muscle fiber lengths l_ce depending on joint angle
        Attention: The mapping of actuator_trnid to qvel is only 1:1 because we have only
        slide or hinge joints and no free joint!!! Otherwise you have to build this mapping
        by looking at every single joint type.

        self.lce_x_tensor contains copy of given actions (they represent the actuator position)
        """
        # Repeat position tensor twice, because both muscles are computed from the same actuator position
        # the operation is NOT applied in-place to the original tensor, only the result is repeated.
        self.lce_tensor = torch.add(torch.mul(actuator_pos.repeat(1, 2), self.moment), self.lce_ref)


    