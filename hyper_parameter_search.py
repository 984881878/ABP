# import trainMlpModel
# import train
# import argparse
#
#
# def search(_weight, _model):
#     if _model == 'AbpCnn':
#         train = train.train
#         lr = 0.05
#         momentum = 0.9
#         train_batch_size = 64
#         eval_batch_size = 128
#     elif _model == 'MlpModel':
#         train = trainMlpModel.train
#         lr = 0.09
#         momentum = 0.9
#         train_batch_size = 50
#         eval_batch_size = 500
#     else:
#         train = None
#         lr = None
#         momentum = None
#         train_batch_size = None
#         eval_batch_size = None
#
#     weight2score = {}
#     for __weight in _weight:
#         for i in range(3):
#             acc = train(20, f'{__weight}-{i}', lr, momentum, __weight, train_batch_size, eval_batch_size)
#             weight2score[f'{__weight}-{i}'] = acc.data
#         _ = train(50, f'{__weight}-50 epochs', lr, momentum, __weight, train_batch_size, eval_batch_size)
#     print(weight2score)
#     with open(f'{_model}-hyper2score.txt', 'w') as f:
#         for items in weight2score.items():
#             f.writelines(f'weight-number: {items[0]}, accuracy: {items[1]:.4f}\n')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', type=str, default=None, choices=['AbpCnn', 'MlpModel'])
#     args = parser.parse_args()
#
#     if args.model is None:
#         raise ValueError('--model [str] is required.')
#
#     weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     search(weight, args.model)
