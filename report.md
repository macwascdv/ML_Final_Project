# Raport z projektu Machine Learning
### Skład zespołu: Marcin Mulawa, Maciej Wąsiatycz
### Kierunek: Data Science, Stacjonarne

## EDA

#### Po sprawdzeniu danych doszliśmy do wniosku, iż różnią się w skali
#### Stwierdziliśmy brak skośności, ale zauważyliśmy obecność outlierów, co skłoniło nas do użycia StandardScalera
#### Wykresy wykazały brak podziału na klastry

## Metryka

#### Ze względu na niezbilansowany dataset, wybraliśmy metrykę f1-weighted, ponieważ zarówno recall, jak i precision są dla nas ważne

## Baseline
#### Do stworzenia baseline'a użyliśmy dwóch modeli - DummyClassifier oraz DecisionTreeClassifier.
#### Dummy Classifier dał wynik f1-weighted równy 0.81


## GridSearch

#### Do gridsearch podaliśmy poniższe parametry:
#### Scaler: StandardScaler
#### Decomposition: PCA i KernelPCA
#### Estimator: DecisionTreeClassifier, SVC i Linear Regression

## Model
#### Model ze względu na swój rozmiar zamieściliśmy na dysku google: https://drive.google.com/drive/folders/1uZP6S-ZArOIAFdz441QIw2YjQyjA0Dfg?usp=sharing
#### Testując go, należy wrzucić plik do folderu ze skryptem final_model
#### Gridsearch wykazał, że najlepszym pipelinem będzie: StandardScaler() KernelPCA oraz Linear Regression 
#### Udało nam się uzyskać wynik 0.93

# Podsumowanie
#### F1-weighted nie okazał się najlepszą metryką, przy ponownej próbie użylibyśmy f2 score
