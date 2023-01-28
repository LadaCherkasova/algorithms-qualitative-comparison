#  5.1 Сравнение алгоритмов 
https://arxiv.org/pdf/1802.03426.pdf

Исследуются и сравниваются алгоритмы уменьшения размерности PCA, UMAP, SVD и tSNE. 
Рассматриваемые датасеты: MNIST, Fashion MNIST, Shuttle и PenDigits

### Импорт библиотек 
```
import matplotlib.pyplot as plt
import umap
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.datasets import fetch_openml, load_digits
```
### Загрузка датасетов 
```
mnist, mnist_labels = fetch_openml(data_id=554, return_X_y=True, as_frame=False)
fmnist, fmnist_labels = fetch_openml(data_id=40996, return_X_y=True, as_frame=False)
shuttle, shuttle_labels = fetch_openml(data_id=40685, return_X_y=True, as_frame=False)
pendigits, pendigits_labels = load_digits(return_X_y=True, as_frame=False)
```
### Датасеты и алгоритмы  
```
datasets = [[mnist, mnist_labels, 'mnist'],
            [fmnist, fmnist_labels, 'fmnist'],
            [pendigits, pendigits_labels, 'pendigits'],
            [shuttle, shuttle_labels, 'shuttle']]

algorithms = [[PCA(n_components=2), 'PCA'],
           [umap.UMAP(n_components=2), 'UMAP'],
           [TruncatedSVD(n_components=2), 'SVD'],
           [TSNE(n_components=2), 'TSNE']]
```
### Выполнение алгоритмов  
```
fig, axs = plt.subplots(4, 4, figsize=(24, 24))

for i in range(4):
    axs[0, i].set_xlabel(datasets[i][2])
    if i != 3:
        axs[i, 0].set_ylabel(algorithms[i][1])

    axs[0, i].xaxis.set_label_position('top')

for i in range(4):
    for j in range(4):
        data = algorithms[i][0].fit_transform(datasets[j][0])
        axs[i, j].scatter(data[:, 0], data[:, 1], c=datasets[j][1].astype(int))

plt.show()
```
### Результат работы   

### Заключение
Точность tSNE алгоритма лучше чем у алгоритмов PCA и SVD, но он уступает им по скорости. UMAP алгоритм является более оптимальным алгоритмом, учитывая критерии точности и времени работы.

