from pathlib import Path
config_dir = Path(__file__).resolve().parent
bev_dir = config_dir.parent
BEV_dir =bev_dir.parent
Root_dir = BEV_dir.parent
data_NCLT_dir = "/home/wyf/data/NCLT"  #请将路径改成NCLT数据集路径
data_oxford_dir = "/home/wyf/data/oxford"  #请将路径改成oxford数据集路径
data_kitti_dir = "/home/wyf/data/kitti"  #请将路径改成kitti数据集路径

__all__ = ['bev_dir', 'BEV_dir', 'Root_dir', 'data_NCLT_dir', 'data_oxford_dir', 'data_kitti_dir']