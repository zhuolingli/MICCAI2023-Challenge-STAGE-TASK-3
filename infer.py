import torch
import pandas as pd
import torchvision.transforms as trans
from lib.datasets.sub3 import STAGE_dataset
import torch.nn.functional as F
from lib.loss import OrdinalRegressionLoss
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    best_model_path = "models/resnetx50/best_model_0.4300.pth"
    testset_root = 'data/STAGE_validation/validation_images'
    aux_info_file = 'data/STAGE_validation/data_info_validation.xlsx'

    model = torch.load(best_model_path)
    model = model.cuda()
    model.eval()

    oct_test_transforms = trans.Compose([
        trans.ToTensor(),
    ])
    
    test_dataset = STAGE_dataset(dataset_root=testset_root,
                                oct_transforms=oct_test_transforms,
                                aux_info_file=aux_info_file,
                                mode='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False)
    cache = []
    criterion = OrdinalRegressionLoss(5).cuda()
    with torch.no_grad():
        for oct_imgs, info_ids, idx in tqdm(test_loader):
            info_ids = info_ids.cuda()  # 同理
            oct_imgs = oct_imgs.cuda()  # 同理
            logits = model(oct_imgs, info_ids)
            loss, like_hoods = criterion(logits.reshape(-1,1), None)
            pre = like_hoods.detach().cpu().numpy().argmax(1).tolist()
            pre = [pre[i] for i in range(0, len(pre), 1)]
            item = [int(idx[0])]
            item.extend(pre)
            cache.append(item)
    cache.sort(key=lambda x:x[0])
    suffix_name =  best_model_path[ -8:-4] + '_' + 'PD_Results' + ".csv"
    result_file = "./final_results/pred_" + suffix_name
    head = ['ID'] + ['point' + str(i) for i in range(1, 53)]
    
    submission_result = pd.DataFrame(cache, columns=head)
    submission_result.to_csv(
        result_file,
        index=False)

if __name__ == '__main__':
    # root = 'data/STAGE_validation/validation_images'
    # sub_list = os.listdir(root)
    # for i in sub_list:
    #     path = os.path.join(root, i, '.DS_Store')
    #     try:
    #         os.remove(path)
    #         print(path)
    #     except:
    #         pass
    main()
