library(MASS) # lda
library(caret) # confusionMatrix
library(pROC) # auc
library(verification) # roc.area, roc.plot (opcional)
library(RColorBrewer) # brewer.pal (opcional)
library(lattice) # levelplot (opcional)

datos <- read.csv("breastcancer.csv")
str(datos)

# Eliminar no predictores (id y última columna vacía)
df <- datos[, -c(1, ncol(datos))]
# Convertir diagnosis (B y M) a los valores 0 y 1, respectivamente, y convertirlo a factor.
df$diagnosis <- as.factor(ifelse(df$diagnosis=="M", "1", "0"))
# Normalizar los datos (predictores o features) para evitar problemas de diferente variabilidad
df[-1] <- apply(df[-1], MARGIN = 2, scale) # scale centra y estandariza por defecto
rm(datos)
dim(df)
head(df)

pca <- prcomp(df[,-1], center = TRUE, scale. = TRUE)
str(pca)

screeplot(pca, type = "lines")

# representa la varianza e)xplicada acumulada
plot(cumsum(pca$sdev^2/sum(pca$sdev^2)), type = "b", xlab = "Número de componentes", ylab = "Varianza explicada acumulada")