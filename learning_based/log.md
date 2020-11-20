## Log

#### 2020-11-20
Работает grid_search.py (ревизия 1fa8cc1366985eab86fe2f97618ff6588079ed5f)
`python3 grid_search.py --x-size 16 --y-size 64 --snr-min 2 --snr-max 7 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 10000 --mod QAM_4  --test-batch-size 5000 --test-every 100`
Зафиксирован linear MMNet_iid, перебираю всевозможные denoiser-ы

#### 2020-11-19
1. Работает запуск offlineTraining (ревизия 9672acced33b31b217d7edda36711dbbc9902245)
`python3 offlineTraining.py  --x-size 32 --y-size 64 --snr-min 18 --snr-max 23 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 10000 --mod QAM_64  --test-batch-size 5000 --linear MMNet_iid --denoiser MMNet --test-every 100` 
2. Обучила также как в статье batch-size 500, 10000 итераций. В первых двух MMNet у меня даже чуть лучше вышло в хвосте, в последнем чуть хуже. MMSE везде идеально совпал.
![transparent_graph](graphs/2020-11-19_16-64-transparent.jpg)

#### 2020-10-05
1. Научились генерить входные данные через `generate1.m` (то же, что и в `generate.m`, только размерности поменьше. Исходный `generate.m` занимал 32 гига памяти и падал)
2. Форкнули репозиторий
3. Переписали чтение данных. Родное не работало, хотело `np.load` делать из директории (но что за директория?)
4. Уперлись в ошибку несоответствия размерностей

Запускать так:
`python3 onlineTraining.py  --x-size 1 --y-size 8 --snr-min 10 --snr-max 15 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 1000 --mod QAM_4  --test-batch-size 5000 --linear MMNet  --denoiser MMNet --data --channels-dir ../../channel_sequences.hdf5 --output-dir output --saveas model --num-channel-samples 10`

Ошибка:
```
ITERATION 0
Feed dict: {<tf.Tensor 'batch_size:0' shape=() dtype=int32>: 500, <tf.Tensor 'lr:0' shape=() dtype=float32>: 0.001, <tf.Tensor 'snr_db_max:0' shape=() dtype=float32>: 15.0, <tf.Tensor 'snr_db_min:0' shape=() dtype=float32>: 10.0}
Train data shape: (500, 256, 128, 8, 10)
Traceback (most recent call last):
  File "onlineTraining.py", line 67, in <module>
    sess.run(train, feed_dict)
  File "/usr/local/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 958, in run
    run_metadata_ptr)
  File "/usr/local/lib/python3.7/site-packages/tensorflow/python/client/session.py", line 1157, in _run
    (np_val.shape, subfeed_t.name, str(subfeed_t.get_shape())))
ValueError: Cannot feed value of shape (500, 256, 128, 8, 10) for Tensor 'H:0', which has shape '(?, 16, 2)'

```

#### To be continued...