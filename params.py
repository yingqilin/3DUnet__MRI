import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--x_roi_size', type=int, default=8)
parser.add_argument('--y_roi_size', type=int, default=30)
parser.add_argument('--z_roi_size', type=int, default=96)
parser.add_argument('--x_padding', type=int, default=10)
parser.add_argument('--y_padding', type=int, default=30)
parser.add_argument('--z_padding', type=int, default=128)
parser.add_argument('--rotDegree', type=float, default=100)
parser.add_argument('--do_rotate', type=bool, default=True)
parser.add_argument('--do_flip', type=bool, default=True)
parser.add_argument('--upper', type=int, default=256)

parser.add_argument('--save_path', type=str, default='E:\\2022spring\Semester Project\code_for_github')
parser.add_argument('--folder_path', type=str, default='E:\\2022spring\Semester Project\code_for_github')
parser.add_argument('--data_path', type=str, default='E:\\2022spring\Semester Project\\bmicdataset')
parser.add_argument('--val_path', type=str, default='E:\\2022spring\Semester Project\\val_dataset')
parser.add_argument('--test_path', type=str, default='E:\\2022spring\Semester Project\\test_dataset')

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=200)

parser.add_argument('--cut_size', type=int, default=128)
parser.add_argument('--cut_stride', type=int, default=24)
parser.add_argument('--num_labels', type=int, default=2)


args = parser.parse_args()