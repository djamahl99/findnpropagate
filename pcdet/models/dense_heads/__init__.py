from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .voxelnext_head import VoxelNeXtHead
from .transfusion_head import TransFusionHead
from .transfusion_head_am import TransFusionHeadAM

# from .transfusion_frustum_head import TransFusionFrustumHead
# from .transfusion_head_sampling import TransFusionHeadSampling
# from .transfusion_head_anchor_matching import TransFusionHeadAnchorMatching
# from .transfusion_head_anchor_matching_learnt_lwh import TransFusionHeadAnchorMatchingLearntVectors
# from .transfusion_head_discrete_iou import TransFusionHeadDiscreteIOU
# from .transfusion_head_gaussian_matching import TransFusionHeadGaussianMatching
# from .transfusion_head_clip import TransFusionHeadCLIP
# from .transfusion_head_2D_proposals import TransFusionHead2DProposals
from .frustum_pointnet_v1 import FrustumPointNetHead
from .frustum_vit_head import FrustumViTHead
from .frustum_proposals import FrustumProposer
from .frustum_proposals_v1 import FrustumProposerOG
from .frustum_proposals_v1_kitti import FrustumProposerOGKITTI
from .frustum_proposals_seg import FrustumProposerSEG
from .frustum_cluster_proposals import FrustumClusterProposer
from .clip_box_classification import CLIPBoxClassification
from .clip_box_cls_maskclip import CLIPBoxClassificationMaskCLIP
from .glip_box_classification import GLIPBoxClassification
from .point_head_box_w_pseudo import PointHeadBoxWPseudos
from .pseudo_processor import PseudoProcessor
from .clip2scene_proposals import CLIP2SceneProposer
from .clip2scene_cc_proposals import CLIP2SceneCCProposer
from .gt_proposals import GTProposals
from .frustum_dbscan import FrustumDBSCAN
from .frustum_ov3ddet import FrustumOV3DET
from .fgr import FGR
__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'CLIPBoxClassification': CLIPBoxClassification,
    'GLIPBoxClassification': GLIPBoxClassification,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'PointHeadBoxWPseudos': PointHeadBoxWPseudos,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'FrustumProposer': FrustumProposer,
    'FrustumProposerOG': FrustumProposerOG,
    'FrustumProposerSEG': FrustumProposerSEG,
    'FrustumProposerOGKITTI': FrustumProposerOGKITTI,
    'FrustumClusterProposer': FrustumClusterProposer,
    'PseudoProcessor': PseudoProcessor,
    'FrustumViTHead': FrustumViTHead,
    'FrustumPointNetHead': FrustumPointNetHead,
    'VoxelNeXtHead': VoxelNeXtHead,
    'TransFusionHead': TransFusionHead,
    'TransFusionHeadAM': TransFusionHeadAM,
    'CLIPBoxClassificationMaskCLIP': CLIPBoxClassificationMaskCLIP,
    'CLIP2SceneProposer': CLIP2SceneProposer,
    'CLIP2SceneCCProposer': CLIP2SceneCCProposer,
    'GTProposals': GTProposals,
    'FrustumDBSCAN': FrustumDBSCAN,
    'FGR': FGR,
    'FrustumOV3DET': FrustumOV3DET
    # 'TransFusionFrustumHead': TransFusionFrustumHead,
    # 'TransFusionHeadCLIP': TransFusionHeadCLIP,
    # 'TransFusionHeadSampling': TransFusionHeadSampling,
    # 'TransFusionHeadAnchorMatching': TransFusionHeadAnchorMatching,
    # 'TransFusionHeadAnchorMatchingLearntVectors': TransFusionHeadAnchorMatchingLearntVectors,
    # 'TransFusionHeadDiscreteIOU': TransFusionHeadDiscreteIOU,
    # 'TransFusionHeadGaussianMatching': TransFusionHeadGaussianMatching,
    # 'TransFusionHead2DProposals': TransFusionHead2DProposals
}
