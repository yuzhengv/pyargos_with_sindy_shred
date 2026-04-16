# Metrics considered

## BA results

### Forecater model performance

```python
Original dataframe shape: 150 rows
Filtered out 43 rows based on the recon rmse of the network framework.
Filtered dataframe shape: 107 rows

Filtered dataframe shape before: 107 rows
Filtered out 26 rows whose indices are not in retained_file_numbers_on_recon_rmse.
Filtered dataframe shape after: 81 rows

MSE DataFrame:
             MSE for dimension 0  MSE for dimension 1  MSE for dimension 2  \
file_number
2                       0.245625             0.239676             0.295966
3                       0.235971             0.204459             0.243518
5                       0.221253             0.188214             0.180009
6                       0.837511             0.392269             0.564551
13                      0.689645             0.602549             0.520857

             Overall_MSE
file_number
2               0.260422
3               0.227983
5               0.196492
6               0.598110
13              0.604350

Summary of MSE DataFrame:
Shape: (81, 4)

Column Names: ['MSE for dimension 0', 'MSE for dimension 1', 'MSE for dimension 2', 'Overall_MSE']

Descriptive Statistics:
       MSE for dimension 0  MSE for dimension 1  MSE for dimension 2  \
count            81.000000            81.000000            81.000000
mean              0.270976             0.280653             0.236636
std               0.207460             0.192691             0.144420
min               0.012789             0.020584             0.012704
25%               0.094354             0.120604             0.148344
50%               0.234962             0.239676             0.216196
75%               0.388910             0.390151             0.295966
max               0.843053             0.948386             0.697196

       Overall_MSE
count    81.000000
mean      0.262755
std       0.167342
min       0.017976
25%       0.118209
50%       0.227612
75%       0.352565
max       0.714476

Missing Values:
MSE for dimension 0    0
MSE for dimension 1    0
MSE for dimension 2    0
Overall_MSE            0
dtype: int64

Average MSE by Column:
MSE for dimension 0: 0.270976
MSE for dimension 1: 0.280653
MSE for dimension 2: 0.236636
Overall_MSE: 0.262755
```

### Rconstrcution via forecaster metrics:

```python
Forecaster Metrics DataFrame:
             MSE_0_50  RMSE_normalized_0_50  MSE_100_200  \
file_number
2            0.664332              0.045434     1.096252
3            0.516366              0.040056     0.760787
5            0.522301              0.040286     0.843165
6            1.211901              0.061365     2.162804
13           0.574677              0.042257     0.965028

             RMSE_normalized_100_200  MSE_200_275  RMSE_normalized_200_275
file_number
2                           0.057989     1.935691                 0.076406
3                           0.048309     1.274821                 0.062006
5                           0.050857     1.454157                 0.066224
6                           0.081452     3.232425                 0.098736
13                          0.054408     1.928242                 0.076259

Shape: (81, 6)
Column Names: ['MSE_0_50', 'RMSE_normalized_0_50', 'MSE_100_200', 'RMSE_normalized_100_200', 'MSE_200_275', 'RMSE_normalized_200_275']

Descriptive Statistics:
        MSE_0_50  RMSE_normalized_0_50  MSE_100_200  RMSE_normalized_100_200  \
count  81.000000             81.000000    81.000000                81.000000
mean    0.910428              0.051314     1.121099                 0.054974
std     0.740819              0.014082     1.120987                 0.020544
min     0.475846              0.038452     0.450260                 0.037164
25%     0.625329              0.044080     0.570419                 0.041830
50%     0.718481              0.047249     0.769528                 0.048585
75%     0.976351              0.055080     1.203958                 0.060771
max     6.664485              0.143904     8.226727                 0.158857

       MSE_200_275  RMSE_normalized_200_275
count    81.000000                81.000000
mean      1.781030                 0.068635
std       1.620208                 0.025864
min       0.578179                 0.041758
25%       0.790877                 0.048839
50%       1.285350                 0.062262
75%       1.935691                 0.076406
max      10.052714                 0.174122

Missing Values:
MSE_0_50                   0
RMSE_normalized_0_50       0
MSE_100_200                0
RMSE_normalized_100_200    0
MSE_200_275                0
RMSE_normalized_200_275    0
dtype: int64

Average by Column:
  MSE_0_50: 0.910428
  RMSE_normalized_0_50: 0.051314
  MSE_100_200: 1.121099
  RMSE_normalized_100_200: 0.054974
  MSE_200_275: 1.781030
  RMSE_normalized_200_275: 0.068635
```

## SINDy Results

### Forecater model performance

```python
Original dataframe shape: 150 rows
Filtered out 43 rows based on the recon rmse of the network framework.
Filtered dataframe shape: 107 rows

Filtered dataframe shape before: 107 rows
Filtered out 43 rows whose indices are not in retained_file_numbers_on_recon_rmse.
Filtered dataframe shape after: 64 rows

MSE DataFrame:
             MSE for dimension 0  MSE for dimension 1  MSE for dimension 2  \
file_number
0                       0.427539             0.299108             0.254279
2                       0.335770             0.308075             0.295043
5                       0.443119             0.337503             0.320428
6                       0.162348             0.320549             0.244168
10                      0.792140             0.694517             0.810304

             Overall_MSE
file_number
0               0.326975
2               0.312963
5               0.367017
6               0.242355
10              0.765654

Summary of MSE DataFrame:
Shape: (64, 4)

Column Names: ['MSE for dimension 0', 'MSE for dimension 1', 'MSE for dimension 2', 'Overall_MSE']

Descriptive Statistics:
       MSE for dimension 0  MSE for dimension 1  MSE for dimension 2  \
count            64.000000            64.000000            64.000000
mean              0.352436             0.340001             0.308068
std               0.211711             0.210636             0.206521
min               0.009910             0.012867             0.007309
25%               0.200338             0.203517             0.164735
50%               0.327331             0.318003             0.259905
75%               0.475985             0.447247             0.434935
max               0.883798             0.965632             0.945599

       Overall_MSE
count    64.000000
mean      0.333501
std       0.186903
min       0.013902
25%       0.221172
50%       0.312030
75%       0.453426
max       0.809298

Missing Values:
MSE for dimension 0    0
MSE for dimension 1    0
MSE for dimension 2    0
Overall_MSE            0
dtype: int64

Average MSE by Column:
MSE for dimension 0: 0.352436
MSE for dimension 1: 0.340001
MSE for dimension 2: 0.308068
Overall_MSE: 0.333501
```

### Rconstrcution via forecaster metrics:

```python

Forecaster Metrics DataFrame:
             MSE_0_50  RMSE_normalized_0_50  MSE_100_200  \
file_number
0            1.242751              0.062141     1.401827
2            0.678044              0.045901     2.465837
5            0.619581              0.043877     1.246088
6            2.429021              0.086877     5.088541
10           1.873959              0.076308     3.760735

             RMSE_normalized_100_200  MSE_200_275  RMSE_normalized_200_275
file_number
0                           0.065575     4.552536                 0.117176
2                           0.086971     4.780824                 0.120078
5                           0.061825     2.300776                 0.083301
6                           0.124937     7.073505                 0.146059
10                          0.107406     6.799331                 0.143201

Shape: (64, 6)
Column Names: ['MSE_0_50', 'RMSE_normalized_0_50', 'MSE_100_200', 'RMSE_normalized_100_200', 'MSE_200_275', 'RMSE_normalized_200_275']

Descriptive Statistics:
        MSE_0_50  RMSE_normalized_0_50  MSE_100_200  RMSE_normalized_100_200  \
count  64.000000             64.000000    64.000000                64.000000
mean    0.993718              0.054263     1.742639                 0.068151
std     0.487172              0.012065     1.431721                 0.026686
min     0.551702              0.041404     0.451068                 0.037197
25%     0.659328              0.045260     0.727201                 0.047230
50%     0.822245              0.050545     1.236461                 0.061586
75%     1.130984              0.059276     2.251999                 0.083100
max     2.636339              0.090508     6.293877                 0.138948

       MSE_200_275  RMSE_normalized_200_275
count    64.000000                64.000000
mean      3.008280                 0.088606
std       2.394833                 0.035230
min       0.606978                 0.042786
25%       1.273910                 0.061970
50%       2.384049                 0.084795
75%       4.118511                 0.111445
max      11.186585                 0.183680

Missing Values:
MSE_0_50                   0
RMSE_normalized_0_50       0
MSE_100_200                0
RMSE_normalized_100_200    0
MSE_200_275                0
RMSE_normalized_200_275    0
dtype: int64

Average by Column:
  MSE_0_50: 0.993718
  RMSE_normalized_0_50: 0.054263
  MSE_100_200: 1.742639
  RMSE_normalized_100_200: 0.068151
  MSE_200_275: 3.008280
  RMSE_normalized_200_275: 0.088606

```

## Comparsion between BA and SINDy

SST 

### Comparing the prediction via forecaster in the latent space

- BA
  Original dataframe shape: 150 rows
  Filtered out 43 rows based on the recon rmse of the network framework.
  Filtered dataframe shape: 107 rows

Filtered dataframe shape before: 107 rows
Filtered out 26 rows whose indices are not in retained_file_numbers_on_recon_rmse.
Filtered dataframe shape after: 81 rows

MSE for dimension 0: 0.270976
MSE for dimension 1: 0.280653
MSE for dimension 2: 0.236636
Overall_MSE: 0.262755

- SINDy

Original dataframe shape: 150 rows
Filtered out 43 rows based on the recon rmse of the network framework.
Filtered dataframe shape: 107 rows

Filtered dataframe shape before: 107 rows
Filtered out 43 rows whose indices are not in retained_file_numbers_on_recon_rmse.
Filtered dataframe shape after: 64 rows

Average MSE by Column:
MSE for dimension 0: 0.352436
MSE for dimension 1: 0.340001
MSE for dimension 2: 0.308068
Overall_MSE: 0.333501

### Comparing the reconstruction via forecaster metrics

- BA
  Average by Column:
  MSE_0_50: 0.910428
  RMSE_normalized_0_50: 0.051314
  MSE_100_200: 1.121099
  RMSE_normalized_100_200: 0.054974
  MSE_200_275: 1.781030
  RMSE_normalized_200_275: 0.068635

- SINDy
  Average by Column:
  MSE_0_50: 0.993718
  RMSE_normalized_0_50: 0.054263
  MSE_100_200: 1.742639
  RMSE_normalized_100_200: 0.068151
  MSE_200_275: 3.008280
  RMSE_normalized_200_275: 0.088606
