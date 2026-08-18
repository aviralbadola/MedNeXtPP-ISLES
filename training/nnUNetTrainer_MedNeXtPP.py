from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from model.create_mednextpp import create_mednextpp


class nnUNetTrainer_MedNeXtPP(nnUNetTrainer):
    """
    nnU-Net v2 trainer for MedNeXt++.

    Uses the MedNeXt++ B configuration with:
    - kernel size = 3
    - deep supervision enabled
    """

    MODEL_ID = "B"

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        return create_mednextpp(
            num_input_channels=num_input_channels,
            num_classes=num_output_channels,
            model_id=nnUNetTrainer_MedNeXtPP.MODEL_ID,
            kernel_size=3,
            deep_supervision=enable_deep_supervision,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Enable or disable deep supervision for MedNeXt++.
        """
        if hasattr(self.network, "do_ds"):
            self.network.do_ds = enabled
