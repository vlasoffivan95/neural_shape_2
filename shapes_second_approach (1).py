

!apt install libcairo2-dev pkg-config python3-dev

!pip install pycairo

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def warn(*args, **kwargs):
    pass

warnings.warn = warn

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, MaxPooling2D, Flatten
# %matplotlib inline

import cairo
num_imgs = 50000

img_size = 32
min_object_size = 4
max_object_size = 16
num_objects = 2

bboxes = np.zeros((num_imgs, num_objects, 4))
imgs = np.zeros((num_imgs, img_size, img_size, 4), dtype=np.uint8)  # в формате BGRA
shapes = np.zeros((num_imgs, num_objects), dtype=int)
num_shapes = 3
shape_labels = ['rectangle', 'circle', 'triangle']
colors = np.zeros((num_imgs, num_objects), dtype=int)
num_colors = 3
color_labels = ['r', 'g', 'b']

for i_img in range(num_imgs):
    surface = cairo.ImageSurface.create_for_data(imgs[i_img], cairo.FORMAT_ARGB32, img_size, img_size)
    cr = cairo.Context(surface)

    # Фон белый
    cr.set_source_rgb(1, 1, 1)
    cr.paint()
    
    # Рисуем фигуры в случайном порядке
    for i_object in range(num_objects):
        shape = np.random.randint(num_shapes)
        shapes[i_img, i_object] = shape
        if shape == 0:  # прямоугольник
            w, h = np.random.randint(min_object_size, max_object_size, size=2)
            x = np.random.randint(0, img_size - w)
            y = np.random.randint(0, img_size - h)
            bboxes[i_img, i_object] = [x, y, w, h]
            cr.rectangle(x, y, w, h)            
        elif shape == 1:  # круг   
            r = 0.5 * np.random.randint(min_object_size, max_object_size)
            x = np.random.randint(r, img_size - r)
            y = np.random.randint(r, img_size - r)
            bboxes[i_img, i_object] = [x - r, y - r, 2 * r, 2 * r]
            cr.arc(x, y, r, 0, 2*np.pi)
        elif shape == 2:  # четырехугольник
            w, h = np.random.randint(min_object_size, max_object_size, size=2)
            x = np.random.randint(0, img_size - w)
            y = np.random.randint(0, img_size - h)
            bboxes[i_img, i_object] = [x, y, w, h]
            cr.move_to(x, y)
            cr.line_to(x+w, y)
            cr.line_to(x+w, y+h)
            cr.line_to(x, y)
            cr.close_path()
        
        color = np.random.randint(num_colors)
        colors[i_img, i_object] = color
        max_offset = 0.3
        r_offset, g_offset, b_offset = max_offset * 2. * (np.random.rand(3) - 0.5)
        if color == 0:
            cr.set_source_rgb(1-max_offset+r_offset, 0+g_offset, 0+b_offset)
        elif color == 1:
            cr.set_source_rgb(0+r_offset, 1-max_offset+g_offset, 0+b_offset)
        elif color == 2:
            cr.set_source_rgb(0+r_offset, 0-max_offset+g_offset, 1+b_offset)
        cr.fill()
        
imgs = imgs[..., 2::-1]  # BGRA переводим RGB

imgs.shape, bboxes.shape, shapes.shape, colors.shape

i = 5
plt.imshow(imgs[i], interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
for bbox, shape, color in zip(bboxes[i], shapes[i], colors[i]):
    plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='k', fc='none'))
    plt.annotate(shape_labels[shape], (bbox[0], bbox[1] + bbox[3] + 0.7), color=color_labels[color], clip_on=False)

X = (imgs - 128.) / 255.
X.shape, np.mean(X), np.std(X)

colors_onehot = np.zeros((num_imgs, num_objects, num_colors))
for i_img in range(num_imgs):
    for i_object in range(num_objects):
        colors_onehot[i_img, i_object, colors[i_img, i_object]] = 1

shapes_onehot = np.zeros((num_imgs, num_objects, num_shapes))
for i_img in range(num_imgs):
    for i_object in range(num_objects):
        shapes_onehot[i_img, i_object, shapes[i_img, i_object]] = 1
        
y = np.concatenate([bboxes / img_size, shapes_onehot, colors_onehot], axis=-1).reshape(num_imgs, -1)
y.shape, np.all(np.argmax(colors_onehot, axis=-1) == colors)

i = int(0.8 * num_imgs)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = imgs[i:]
test_bboxes = bboxes[i:]

filter_size = 3
pool_size = 2


model = Sequential([
        Convolution2D(32, 6, 6, input_shape=X.shape[1:], dim_ordering='tf', activation='relu'), 
        MaxPooling2D(pool_size=(pool_size, pool_size)), 
        Convolution2D(64, filter_size, filter_size, dim_ordering='tf', activation='relu'), 
        MaxPooling2D(pool_size=(pool_size, pool_size)), 
        Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'), 
        Convolution2D(128, filter_size, filter_size, dim_ordering='tf', activation='relu'), 
        Flatten(), 
        Dropout(0.4), 
        Dense(256, activation='relu'), 
        Dropout(0.4), 
        Dense(y.shape[-1])
    ])

model.compile('adadelta', 'mse')

# Переворачиваем bbox во время тренировок
def IOU(bbox1, bbox2):
    '''Рассчитываем перекрытие между двумя ограничивающими прямоугольниками [x, y, w, h] как площадь пересечения над областью единицы'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # без перекрытия
        return 0
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U

def dist(bbox1, bbox2):
    return np.sqrt(np.sum(np.square(bbox1[:2] - bbox2[:2])))

num_epochs_flipping = 20
num_epochs_no_flipping = 0  # не оказывает существенного влияния

flipped_train_y = np.array(train_y)
flipped = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
ious_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
dists_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
mses_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
acc_shapes_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))
acc_colors_epoch = np.zeros((len(train_y), num_epochs_flipping + num_epochs_no_flipping))

flipped_test_y = np.array(test_y)
flipped_test = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
ious_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
dists_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
mses_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
acc_shapes_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))
acc_colors_test_epoch = np.zeros((len(test_y), num_epochs_flipping + num_epochs_no_flipping))


for epoch in range(num_epochs_flipping):
    print('Epoch', epoch)
    model.fit(train_X, flipped_train_y, nb_epoch=1, validation_data=(test_X, test_y), verbose=2)
    pred_y = model.predict(train_X)

    for sample, (pred, exp) in enumerate(zip(pred_y, flipped_train_y)):
        
        pred = pred.reshape(num_objects, -1)
        exp = exp.reshape(num_objects, -1)
        
        pred_bboxes = pred[:, :4]
        exp_bboxes = exp[:, :4]
        
        ious = np.zeros((num_objects, num_objects))
        dists = np.zeros((num_objects, num_objects))
        mses = np.zeros((num_objects, num_objects))
        for i, exp_bbox in enumerate(exp_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                ious[i, j] = IOU(exp_bbox, pred_bbox)
                dists[i, j] = dist(exp_bbox, pred_bbox)
                mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))
                
        new_order = np.zeros(num_objects, dtype=int)
        
        for i in range(num_objects):
            # Найдем pred и exp bbox с максимальным значением iou и назначим их друг другу (т. е. переключим положения exp bboxes в y).
            ind_exp_bbox, ind_pred_bbox = np.unravel_index(ious.argmax(), ious.shape)
            ious_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
            dists_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
            mses_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
            ious[ind_exp_bbox] = -1  # установим для iou назначенных bbox значение -1, чтобы они не были назначены снова
            ious[:, ind_pred_bbox] = -1
            new_order[ind_pred_bbox] = ind_exp_bbox
        
        flipped_train_y[sample] = exp[new_order].flatten()
        
        flipped[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))
        ious_epoch[sample, epoch] /= num_objects
        dists_epoch[sample, epoch] /= num_objects
        mses_epoch[sample, epoch] /= num_objects
        
        acc_shapes_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4:4+num_shapes], axis=-1) == np.argmax(exp[:, 4:4+num_shapes], axis=-1))
        acc_colors_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1) == np.argmax(exp[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1))

    
    # Посчитаем метрики на тестовых данных
    pred_test_y = model.predict(test_X)

    for sample, (pred, exp) in enumerate(zip(pred_test_y, flipped_test_y)):
        pred = pred.reshape(num_objects, -1)
        exp = exp.reshape(num_objects, -1)
        
        pred_bboxes = pred[:, :4]
        exp_bboxes = exp[:, :4]
        
        ious = np.zeros((num_objects, num_objects))
        dists = np.zeros((num_objects, num_objects))
        mses = np.zeros((num_objects, num_objects))
        for i, exp_bbox in enumerate(exp_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                ious[i, j] = IOU(exp_bbox, pred_bbox)
                dists[i, j] = dist(exp_bbox, pred_bbox)
                mses[i, j] = np.mean(np.square(exp_bbox - pred_bbox))
                
        new_order = np.zeros(num_objects, dtype=int)
        
        for i in range(num_objects):
            # Найдем pred и exp bbox с максимальным значением iou и назначим их друг другу (т. е. переключим положения exp bboxes в y).
            ind_exp_bbox, ind_pred_bbox = np.unravel_index(mses.argmin(), mses.shape)
            ious_test_epoch[sample, epoch] += ious[ind_exp_bbox, ind_pred_bbox]
            dists_test_epoch[sample, epoch] += dists[ind_exp_bbox, ind_pred_bbox]
            mses_test_epoch[sample, epoch] += mses[ind_exp_bbox, ind_pred_bbox]
            mses[ind_exp_bbox] = 1000000 # -1  # установим для iou назначенных bbox значение -1, чтобы они не были назначены снова
            mses[:, ind_pred_bbox] = 10000000 # -1
            new_order[ind_pred_bbox] = ind_exp_bbox
        
        flipped_test_y[sample] = exp[new_order].flatten()
        
        flipped_test[sample, epoch] = 1. - np.mean(new_order == np.arange(num_objects, dtype=int))
        ious_test_epoch[sample, epoch] /= num_objects
        dists_test_epoch[sample, epoch] /= num_objects
        mses_test_epoch[sample, epoch] /= num_objects
        
        acc_shapes_test_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4:4+num_shapes], axis=-1) == np.argmax(exp[:, 4:4+num_shapes], axis=-1))
        acc_colors_test_epoch[sample, epoch] = np.mean(np.argmax(pred[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1) == np.argmax(exp[:, 4+num_shapes:4+num_shapes+num_colors], axis=-1))
       
            
    print('Flipped {} % of all elements'.format(np.mean(flipped[:, epoch]) * 100.))
    print('Mean IOU: {}'.format(np.mean(ious_epoch[:, epoch])))
    print('Mean dist: {}'.format(np.mean(dists_epoch[:, epoch])))
    print( 'Mean mse: {}'.format(np.mean(mses_epoch[:, epoch])))
    print( 'Accuracy shapes: {}'.format(np.mean(acc_shapes_epoch[:, epoch])))
    print( 'Accuracy colors: {}'.format(np.mean(acc_colors_epoch[:, epoch])))
    
    print( '--------------- TEST ----------------')
    print('Flipped {} % of all elements'.format(np.mean(flipped_test[:, epoch]) * 100.))
    print('Mean IOU: {}'.format(np.mean(ious_test_epoch[:, epoch])))
    print('Mean dist: {}'.format(np.mean(dists_test_epoch[:, epoch])))
    print('Mean mse: {}'.format(np.mean(mses_test_epoch[:, epoch])))
    print('Accuracy shapes: {}'.format(np.mean(acc_shapes_test_epoch[:, epoch])))
    print('Accuracy colors: {}]\n'.format(np.mean(acc_colors_test_epoch[:, epoch])))

# model.layers
weights = model.layers[0].get_weights()[0]
weights = weights.transpose(3, 0, 1, 2)
print(weights.shape)

print(np.mean(weights[0]), np.std(weights[0]), np.min(weights[0]), np.max(weights[0]))
adj_weights = (weights * 255.) + 128.

print(np.mean(adj_weights[0]), np.std(adj_weights[0]), np.min(adj_weights[0]), np.max(adj_weights[0]))
plt.figure(figsize=(16, 8))

for i in range(24):
    plt.subplot(4, 6, i+1)
    plt.imshow(adj_weights[i, :, :], interpolation='none', origin='lower', cmap='Greys')

mean_ious_epoch = np.mean(ious_epoch, axis=0)
mean_dists_epoch = np.mean(dists_epoch, axis=0)
mean_mses_epoch = np.mean(mses_epoch, axis=0)
plt.plot(mean_ious_epoch, label='Mean IOU')  # между предсказанным и истинными bboxes
plt.plot(mean_dists_epoch, label='Mean distance')  # относительно размера изображения
plt.plot(mean_mses_epoch, label='Mean mse')  # относительно размера изображения
plt.annotate(np.round(np.max(mean_ious_epoch), 3), (len(mean_ious_epoch)-1, mean_ious_epoch[-1]+0.03), horizontalalignment='right', color='b')
plt.annotate(np.round(np.min(mean_dists_epoch), 3), (len(mean_dists_epoch)-1, mean_dists_epoch[-1]+0.03), horizontalalignment='right', color='g')
plt.annotate(np.round(np.min(mean_mses_epoch), 3), (len(mean_mses_epoch)-1, mean_mses_epoch[-1]+0.03), horizontalalignment='right', color='r')

# TEST.
mean_ious_epoch = np.mean(ious_test_epoch, axis=0)
mean_dists_epoch = np.mean(dists_test_epoch, axis=0)
mean_mses_epoch = np.mean(mses_test_epoch, axis=0)
plt.plot(mean_ious_epoch, 'b--')  # между предсказанным и истинными bboxes
plt.plot(mean_dists_epoch, 'g--')  # относительно размера изображения
plt.plot(mean_mses_epoch, 'r--')  # относительно размера изображения

plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.ylim(0, 0.6)

mean_acc_shapes_epoch = np.mean(acc_shapes_epoch, axis=0)
mean_acc_colors_epoch = np.mean(acc_colors_epoch, axis=0)
plt.plot(mean_acc_shapes_epoch, label='Accuracy shapes')  # между предсказанным и истинными bboxes
plt.plot(mean_acc_colors_epoch, label='Accuracy colors')
plt.annotate(np.round(np.max(mean_acc_shapes_epoch), 3), (len(mean_acc_shapes_epoch)-1, mean_acc_shapes_epoch[-1]+0.03), horizontalalignment='right', color='b')
plt.annotate(np.round(np.max(mean_acc_colors_epoch), 3), (len(mean_acc_colors_epoch)-1, mean_acc_colors_epoch[-1]+0.03), horizontalalignment='right', color='g')

# TEST.
mean_acc_shapes_epoch = np.mean(acc_shapes_test_epoch, axis=0)
mean_acc_colors_epoch = np.mean(acc_colors_test_epoch, axis=0)
plt.plot(mean_acc_shapes_epoch, 'b--')  # между предсказанным и истинными bboxes
plt.plot(mean_acc_colors_epoch, 'g--')

plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.ylim(0.5, 1)

pred_y = model.predict(test_X)
pred_y = pred_y.reshape(len(pred_y), num_objects, -1)
pred_bboxes = pred_y[..., :4] * img_size
pred_shapes = np.argmax(pred_y[..., 4:4+num_shapes], axis=-1).astype(int)  # взять максимум из вероятностей

pred_colors = np.argmax(pred_y[..., 4+num_shapes:4+num_shapes+num_colors], axis=-1).astype(int)
pred_bboxes.shape, pred_shapes.shape, pred_colors.shape

plt.figure(figsize=(16, 8))
for i_subplot in range(1, 9):
    plt.subplot(2, 4, i_subplot)
    i = np.random.randint(len(test_X))
    plt.imshow(test_imgs[i], interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    for bbox, shape, color in zip(pred_bboxes[i], pred_shapes[i], pred_colors[i]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='k', fc='none'))
        plt.annotate(shape_labels[shape], (bbox[0], bbox[1] + bbox[3] + 0.7), color=color_labels[color], clip_on=False, bbox={'fc': 'w', 'ec': 'none', 'pad': 1, 'alpha': 0.6})