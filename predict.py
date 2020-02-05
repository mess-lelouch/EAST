from model import East
from loss import *
import config as cfg
from utils.save import *
from utils.myzip import *
from eval import predict
from hmean import compute_hmean



def get_model():
    model = nn.DataParallel(East(), device_ids=cfg.gpu_ids).cuda()

    weightpath = os.path.abspath(cfg.checkpoint)

    checkpoint = torch.load(weightpath)

    model.load_state_dict(checkpoint['state_dict'])

    return model


def main():
    model = get_model()
    criterion = LossFunc()
    epoch = 9999

    # create res_file and img_with_box
    output_txt_dir_path = predict(model, criterion, epoch)

    # Zip file
    submit_path = MyZip(output_txt_dir_path, epoch)

    # submit and compute Hmean
    hmean = compute_hmean(submit_path)

    results = {
        'epoch': epoch,
        'output_txt_dir_path': output_txt_dir_path,
        'submit_path': submit_path,
        'hmean': hmean
    }

    print(results)


if __name__ == '__main__':
    main()
