
############

library(readxl)
library(tsfgrnn)
library(nnfor)
library(tidyverse)
library(sidrar)
library(lubridate)
library(forecast)
library(urca)
library(rbcb)
library(tstools)
library(Quandl)
library(scales)
library(gridExtra)
library(caret)
library(vars)


#################################################################
#################   ARIMA     ############################

pnad.raw = get_sidra(api='/t/6318/n1/all/v/1641/p/all/c629/all')

desocupada <- pnad.raw %>%
  filter(`Condição em relação à força de trabalho e condição de ocupação (Código)` == 32446) %>%
  .[-length(pnad.raw),] %>%
  pull(Valor) %>% 
  ts(start=c(2012,03),end = c(2019,12), freq=12)  

forca_de_trabalho <- pnad.raw %>%
  filter(`Condição em relação à força de trabalho e condição de ocupação (Código)` == 32386) %>%
  .[-length(pnad.raw),] %>%
  pull(Valor) %>% 
  ts(start=c(2012,03),end = c(2019,12), freq=12)

desemprego <- desocupada/forca_de_trabalho*100
desemprego <-  ts(desemprego[-1],start = c(2012,04), frequency = 12)


decomposicao <-  decompose(desemprego)
plot(decomposicao)


qqnorm(desemprego)
qqline(desemprego)




## estacionaridade 

teste <-  ur.pp(desemprego)
summary(teste)

ndiffs(desemprego)
desemprego2 <-  diff(desemprego, differences = 2)
teste <-  ur.pp(desemprego2)
summary(teste)


hist(desemprego2)

qqnorm(desemprego)
qqline(desemprego)

qqnorm(desemprego2)
qqline(desemprego2)

dev.off()   
tsdisplay(desemprego2)

Box.test(desemprego2, type = 'Ljung-Box')

train <-  window(desemprego, start = c(2012,04), end= c(2018,12))
test <-  desemprego %>% window(start = c(2019,1))



model <-  auto.arima(train,d= 2, approximation = F, seasonal = F)
model2 <-  auto.arima(train, approximation = F)


checkresiduals(model)
summary(model)

checkresiduals(model2)
summary(model2)




arima_forecast <- forecast(model, h = 12)
arima_forecast2 <-  forecast(model2, h = 12)

p1 <-  autoplot(arima_forecast) + 
  autolayer(test)

p2  <-  autoplot(arima_forecast2)+
  autolayer(test)

grid.arrange(p1,p2)

acc_arima <- accuracy(arima_forecast, test)
acc_arima2 <- accuracy(arima_forecast2, test)


Box.test(residuals(arima_forecast))
Box.test(residuals(arima_forecast2))




################################################
####### FEED FOWARD NEURAL NETWORK ##############

ff_model <-  nnetar(train, p = 40, size = 20 )

checkresiduals(ff_model)

## forecasting 
ff_forecast <-  forecast(ff_model, h= 12)
print(ff_forecast)

autoplot(ff_forecast )+ 
  autolayer(desemprego)

acc_ff <-  accuracy(ff_forecast, desemprego)
acc_ff



####################################################
####    Radial Basis Function ###########
rbf <-  grnn_forecasting(train, h = 12, lags = 1:12) 

autoplot(rbf) + 
  autolayer(test) 


acc_rbf <-  accuracy(rbf$prediction, test)
acc_rbf


summary(rbf)


#######################################################
#######    VECM #################
## Coleta e tratamento dos dados 




##  FBCF PNAD/IBGE
url = 'https://www.ipea.gov.br/cartadeconjuntura/wp-content/uploads/2022/03/220404_cc_53_dados_indicador_ipea_fbcf_dez21.xlsx'
download.file(url, destfile = 'fbcf.xlsx', mode = 'wb')
fbcf_raw <- read_excel('fbcf.xlsx', col_names = T, skip = 1)



fbcf <- fbcf_raw[,5] %>% 
  rename('FBCF' = `Indicador Ipea de FBCF` ) %>% 
  ts(start = c(1996,01),end = c(2019,12), frequency = 12) %>% 
  window(start = c(2012,04))




## IPCA - Série histórica com número índice - Dados do IBGE
ipca <- get_sidra(api='/t/1737/n1/all/v/2266/p/all/d/v2266%2013') %>%
  pull(Valor) %>%
  ts(start=c(1979,12), end = c(2019,12), freq=12) %>%
  window(start=c(2012,04) ) 



## SELIC dados do Banco central
selic <- GetBCBData::gbcbd_get_series(id= 4189,
                                      first.date = '2012-04-01',
                                      last.date = '2019-12-01',
                                      use.memoise = F)  %>% 
  pull(value) %>%
  ts(start = c(2012,04), frequency = 12) 




## Juntando as séries temporais
data_ts <- ts.intersect(desemprego, fbcf, selic, ipca) 

df <-  as.data.frame(data_ts) %>%
  mutate(date = seq(as_date('2012-04-01'), as_date('2019-12-01'), 'month')) %>%
  relocate(date) 



## Amostra de treino e teste
df <-  as.data.frame(data_ts) %>%
  mutate(date = seq(as_date('2012-04-01'), as_date('2019-12-01'), 'month'))
df$date <-  NULL
training.2 = slice(df, -c(82:93))
testing.2 = slice(df, c(82:93))


## Testando cointegração 
d <- VARselect(training.2, lag.max = 12, type = 'both')
d$selection


j.eigen <- ca.jo(training.2, type = 'eigen', K = 4,
                 ecdet = 'const',
                 spec = 'transitory',
                 season = 12)
summary(j.eigen)


## Criando modelo VEC
vec <-  cajorls(j.eigen, r = 3)
summary(vec$rlm)
vec
model.1 = vec2var(j.eigen, r = 3)


# Arch Effects 
arch <-  arch.test(model.1, lags.multi = 12, multivariate.only = T)
arch

# Normalidade dos resíduos 
norm <-  normality.test(model.1)
norm
hist(model.1$resid)
## forecast 
forecast <-  predict(model.1, n.ahead = nrow(testing.2), ci = 0.95)
training <- ts(training.2, start = c(2012,04), freq= 12)
testing <- ts(testing.2, start = c(2019,1), frequency = 12)

f.cast_desemprego <- ts(forecast$fcst$desemprego, start = start(testing),
                        frequency = 12)
autoplot(cbind(f.cast_desemprego, testing[,1]))


## Avaliação
acc_vec <-  accuracy(f.cast_desemprego, testing[,1])
print(xtable::xtable(acc))
acc

## Visualizando 
df.2 = as.data.frame(forecast$fcst$desemprego) %>%
  mutate(date = seq(as_date('2019-01-01'), as_date('2019-12-01'), 'month')) %>%
  relocate(date) %>%
  rename(fitted=fcst)

df.2$testing = testing.2$desemprego



df.2 %>% 
  ggplot(aes(x = date)) + 
  geom_line(aes(y = fitted, colour = 'Forecast'), size = 1.2) + 
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = .2, fill = 'grey') +
  geom_line(aes(y = testing, colour = 'Testing'), size = 1.2) +
  theme(legend.position = c(.1,.2),
        plot.title = element_text(size = 18)) +
  scale_x_date(breaks = date_breaks('1 month'),
               labels = date_format('%m/%Y')) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  labs(x = '', y = '%',
       title = 'Brazil Unemployment Rate Forecast',
       subtitle = 'A Vector Error Correction Forecasting Model')






#############################################################

df3 <- data.frame(df.2$date)
colnames(df3) <- 'date'
df3$vecm <- df.2$fitted
df3$ff <- as.double(ff_forecast$mean)
df3$mlp <-as.double( mlp_forecast2$mean)
df3$rbf <- as.double(rbf$prediction)
df3$arima <-as.double(arima_forecast2$mean)
df3$test <- df.2$testing
df3 <- as_tibble(df3)

str(df3)
df3


df3 %>% 
  ggplot(aes(x = date)) + 
  geom_line(aes(y =vecm, colour = 'VECM'), size = .9) +
  geom_line(aes(y =ff, colour = 'Feed-forward neural networks with a single hidden layer'),
            size = .9) +
  geom_line(aes(y =mlp, colour = 'Multilayer Perceptron'), size = .9) +
  geom_line(aes(y =rbf, colour = 'General regression neural network'), size = .9) +
  geom_line(aes(y =arima, colour = 'SARIMA'), size = .9) +
  geom_line(aes(y =test, colour = 'Unemployment'), size = .9, linetype = 'dashed') +
  theme(legend.position = c(.1,.2),
       plot.title = element_text(size = 18)) +
  scale_x_date(breaks = date_breaks('1 month'),
               labels = date_format('%m/%Y')) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  labs(x = '', y = '%',
       title = 'Brazil Unemployment Rate Forecast')






v1 = df3  %>% 
  ggplot(aes(x = date)) + 
  geom_line(aes(y =test, colour = 'Unemployment'), size = .9) + 
  geom_line(aes(y =ff), size = .9, colour = 'darkblue' ) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        plot.title = element_text(size = 18)) + 
  scale_x_date(breaks = date_breaks('1 month'),
               labels = date_format('%m/%Y')) +
  labs(x = '', y = '%',
       title = 'Feed-forward neural networks with a single hidden layer')


v1


v2 = df3  %>% 
  ggplot(aes(x = date)) + 
  geom_line(aes(y =test, colour = 'Unemployment'), size = .9) + 
  geom_line(aes(y =mlp), size = .9, colour = 'darkblue' ) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        plot.title = element_text(size = 18)) + 
  scale_x_date(breaks = date_breaks('1 month'),
               labels = date_format('%m/%Y')) +
  labs(x = '', y = '%',
       title = 'Multilayer Perceptron')


v2

v3 = df3  %>% 
  ggplot(aes(x = date)) + 
  geom_line(aes(y =test, colour = 'Brazil Unemployment Rate'), size = .9) + 
  geom_line(aes(y =rbf), size = .9, colour = 'darkblue' ) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        plot.title = element_text(size = 18)) + 
  scale_x_date(breaks = date_breaks('1 month'),
               labels = date_format('%m/%Y')) +
  labs(x = '', y = '%',
       title = 'General regression neural network')


v3



v4 = df3  %>% 
  ggplot(aes(x = date)) + 
  geom_line(aes(y =test, colour = 'Brazil Unemployment Rate'), size = .9) + 
  geom_line(aes(y =vecm), size = .9, colour = 'darkblue' ) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        plot.title = element_text(size = 18)) + 
  scale_x_date(breaks = date_breaks('1 month'),
               labels = date_format('%m/%Y')) +
  labs(x = '', y = '%',
       title = 'Vector Error Correction Model ')
v4

v5 <-  df3  %>% 
  ggplot(aes(x = date)) + 
  geom_line(aes(y =test, colour = 'Brazil Unemployment Rate'), size = .9) + 
  geom_line(aes(y =arima), size = .9, colour = 'darkblue' ) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1), 
        plot.title = element_text(size = 18)) + 
  scale_x_date(breaks = date_breaks('1 month'),
               labels = date_format('%m/%Y')) +
  labs(x = '', y = '%',
       title = 'SARIMA ')
v5

grid.arrange(v1,v2,v3,
             layout_matrix = matrix(c(1,2,3),
                                    ncol = 1, byrow = T))


grid.arrange(v3,v5,v4,
             layout_matrix = matrix(c(1,2,3),
                                    ncol = 1, byrow = T))




acv_arima <- as.data.frame(acc_arima2)
acv_rbf <- as.data.frame(acc_rbf)
acv_vec <- as.data.frame(acc_vec)



rmse_rbf <- acv_rbf[,2]
rmse_arima <- acv_arima[,2]
rmse_vec <- acv_vec[,2]

rmse_t <- rbind(rmse_rbf, rmse_arima, rmse_vec)

mape_rbf <- acv_rbf[,5]
mape_arima <- acv_arima[,5]
mape_vec <- acv_vec[,5]

mape_t <-  rbind(mape_rbf, mape_arima, mape_vec)

acc_t <- cbind(rmse_t, mape_t)

acc_t <- as.data.frame(acc_t) %>% 
  rename(RMSE = 'V2',
         MAPE = 'V4') %>% 
  dplyr::select(RMSE, MAPE )

rownames(acc_t) <- c('RBF', 'SARIMA', 'VEC')
grid.arrange(v3,v5,v4,
             layout_matrix = matrix(c(1,2,3),
                                    ncol = 1, byrow = T))
acc_t

