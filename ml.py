import numpy as np
from matplotlib import pyplot as plt 
from matplotlib.colors import ListedColormap
from typing import Literal


# dolar = rq.get(
#     url='https://ve.dolarapi.com/v1/historicos/dolares/oficial'
# )

# historico = dolar.json()

# prom = []


# for i in range(len(historico)):
    
#     prom2 = []
#     for j,n in enumerate(['fuente','promedio','fecha']):

#         prom2.append(historico[i][n])

#     prom.append(prom2)

# x2 = []
# y2 = []
# z2 = []

# dia = 0
# for i in range(len(historico)//2, len(historico)):


#     dia += 1
#     x2.append(dia)
#     z2.append(int(prom[i][2][-2:]))
#     y2.append(np.log(prom[i][1]))

# data = np.column_stack((x2,z2,y2))

class Modelo_base():

    def __init__(self, datos:list[list], estandarizar=True):
        
        pass
    
    

    def _estandarizar(self):

        pass



    def entrenar(self):    

        pass



    def f_hat(self, x):

        pass



    def MSE(self):

        pass

    def R_2(self):

        pass


    def graficar_f(self, puntos=True, final_percent=0.2, precision=100):
        
        pass

class Regresion_polinomial():
    
    
    def __init__(self,grado:int):
        
        self.grado = grado
        
        self._params = []
        
        self._xmean = None
        self._xstd = None
        self._ymean = None
        self._ystd = None


        
        self._estandarizado = False
        self._entrenado = False
    
    

    def _estandarizar(self, datos: list[list]):

        res = []

        self._xmean = np.mean(datos)
        self._xstd = np.std(datos)
        self._ymean = np.mean(datos)
        self._ystd = np.std(datos)

        x = [i[0] for i in datos]
        y = [i[1] for i in datos]

        for i in range(len(datos)):

            res.append( [ (x[i] - self._xmean) / self._xstd , (y[i] - self._ymean) / self._ystd ] )    

        self._estandarizado = True

        return res



    def entrenar(self, datos: list[list], estandarizar=True):    


        if estandarizar == True:
            
            datos = self._estandarizar(datos=datos)
        

        A = np.zeros(shape=( len(datos), self.grado + 1))

        for i,d in enumerate(datos):

            app = [1]

            for j in range(1,self.grado + 1):

                app.append(np.power(d[0], j))


            A[i] = app

        qr = np.linalg.qr(A)

        # transpose = np.linalg.matrix_transpose(A)

        self._params = np.linalg.solve(
            qr.R,
            np.linalg.matmul( np.linalg.matrix_transpose(qr.Q), np.array([i[1] for i in datos]) )
        )

        self._entrenado = True        

        
        # matp = np.zeros(shape=(grado + 1, grado + 1))
        # matr = np.zeros(shape=(grado + 1))


        # for col in range(grado + 1):

        #     for fil in range(grado + 1):

        #         matp[col][fil] += self._sumatoria_x( (grado - col) + (grado - fil) )

        # for fil in range(grado + 1):

        #     matr[fil] += self._sumatoria_xy(grado - fil)



        # qr = np.linalg.qr(matp)

        # self._solv = np.linalg.solve(qr.R, np.linalg.matmul( np.linalg.matrix_transpose(qr.Q) , matr) )

        self._entrenado = True



    def f_hat(self, x):

        if self._entrenado == False:
            raise RuntimeError("No se ha entrenado el modelo")

        if hasattr(x, '__len__'):
            raise RuntimeError("Solo envia una X float :v")

        res = 0


        x = [np.power(x, i) for i in range(1, self.grado + 1)]


        if self._estandarizado == True:

            x_std = (x - self._xmean) / self._xstd

            x1 = np.concatenate(([1.0], x_std))

            res = np.dot( self._params, x1)

            return res * self._ystd + self._ymean


        x1 = np.concatenate(([1.0], x))
        
        res = np.dot( self._params, x1)


        # if self._estandarizado == True:


        #     for i in range(self.grado + 1):

        #         res += self._solv[i] * np.power( (x - self.xmean)/self.xstd, self.grado - i)


        #     return res * self.ystd + self.ymean
        
        
        # for i in range(self.grado + 1):

        #     res += self._solv[i] * np.power( x, self.grado - i)


        return res 



    def MSE(self, y_pred, y):

        error = 0

        for i in range(len(y_pred)):

            error += (y_pred[i] - y[i]) * (y_pred[i] - y[i])


        return error / len(y_pred)

    def R_2(self, y_pred, y):


        total_var = 0
        res_var = 0

        mean_total_var = np.mean(y)

        for i in range(len(y_pred)):

            res_var += (y[i] - y_pred[i]) * (y[i] - y_pred[i])

            total_var += (y[i] - mean_total_var) * (y[i] - mean_total_var)

        return 1 - (res_var/total_var)


    def graficar_f(self, datos: list[list], puntos=True, final_percent=0.2, precision=100):

        x = [i[0] for i in datos]
        y = [i[1] for i in datos]
        
        max_x = np.max(x)
        
        x_plot = np.linspace(0, round(max_x * (1 + final_percent)), precision)
        y_plot = np.array([[self.f_hat(i)] for i in x_plot])
        
        plt.plot(x_plot, y_plot)
        
        if puntos == False:
            plt.plot(x, y)
        else:
            plt.scatter(x, y)
            
        

        plt.grid(True)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)


        plt.show()

class Regresion_lineal_multiple():
    
    
    def __init__(self):

        self._estandarizado = False
        self._entrenado = False
     
        self._params = np.array([])
        # self._dis = np.array([])
        

    
    

    def _estandarizar(self, datos: list[list]):

        res = np.zeros_like(datos)

        self.means = np.mean(datos, axis=0)
        self.stds = np.std(datos, axis=0)

        for i,d in enumerate(datos):

            res[i] = [ (d[j] - self.means[j]) / self.stds[j] for j in range(len(d)) ]

        self._estandarizado = True

        return res



    def entrenar(self, datos: list[list], estandarizar=True):    

        if estandarizar == True:

            datos = self._estandarizar(datos=datos)
        
        
        A = np.zeros(shape=( len(datos),len(datos[0])))

        for i,d in enumerate(datos):

            app = [1]

            for j in range(len(d) - 1):

                app.append(d[j])

            A[i] = app

        qr = np.linalg.qr(A)

        # transpose = np.linalg.matrix_transpose(A)

        self._params = np.linalg.solve(
            qr.R,
            np.linalg.matmul( np.linalg.matrix_transpose(qr.Q), [i[-1] for i in datos] )
        )

        self._entrenado = True



    def f_hat(self, x: list):

        if self._entrenado == False:
            raise AttributeError


        if self._estandarizado == True:

            x_std = (x - self.means[:-1]) / self.stds[:-1]

            x1 = np.concatenate(([1.0], x_std))

            res = np.dot( self._params, x1)

            return res * self.stds[-1] + self.means[-1]


        x1 = np.concatenate(([1.0], x))
        
        res = np.dot( self._params, x1)

            

        return res



    def MSE(self, y_pred, y):

        error = 0


        for i in range(len(y_pred)):

            error += (y_pred[i] - y[i]) * (y_pred[i] - y[i])


        return error / len(y_pred)

    def R_2(self, y_pred, y):


        total_var = 0
        res_var = 0

        mean_total_var = np.mean(y)


        for i in range(len(y_pred)):

            res_var += (y_pred[i] - y[i]) * (y_pred[i] - y[i])

            total_var += (y[i] - mean_total_var) * (y[i] - mean_total_var)


        return 1 - (res_var/total_var)


    def graficar_f(self,datos:list[list], puntos=True, final_percent=0.2, precision=100):

        if len(datos[0]) == 2:

            x_data = [i[0] for i in datos]
            y = [i[1] for i in datos]

            xmax = np.max(x_data)
            
            x_plot = np.linspace(0, round(xmax * (1 + final_percent)), precision)

            x_plot.resize((len(x_plot), len(datos[0]) - 1))

            y_plot = np.array([self.f_hat(x_plot[i]) for i in range(len(x_plot))])

            
            plt.plot(x_plot, y_plot)
            
            if puntos == False:
                plt.plot(x_data, y)
            else:
                plt.scatter(x_data, y)


        elif len(datos[0]) == 3:

            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            x_data = [i[0] for i in datos]
            z_data = [i[1] for i in datos]


            xmax = np.max(x_data)
            ymax = np.max(z_data)
            
            x_plot = np.linspace(0, round(xmax * (1 + final_percent)), precision)

            
            x_plot = x_plot[:, np.newaxis]


            y_plot = np.linspace(0, round(ymax * (1 + final_percent)), precision)

            y_plot = y_plot[:, np.newaxis]


            xy_plot = np.column_stack((x_plot, y_plot))


            z_plot = np.array([self.f_hat(xy_plot[i]) for i in range(len(x_plot))])

            x_vals = np.linspace(min(x_data), max(x_data), 30)
            z_vals = np.linspace(min(z_data), max(z_data), 30)

            X, Z = np.meshgrid(x_vals, z_vals)

            Y = np.array([[self.f_hat([x, z]) for x, z in zip(row_x, row_z)]
                        for row_x, row_z in zip(X, Z)])         

            ax.plot_surface(X, Z, Y, alpha=0.7)
            
            
            if puntos == False:
                ax.plot(x_data, z_data, self.y)
            else:
                ax.scatter(x_data, z_data, self.y)

            ax.grid(True)
            ax.axhline(0, color='black', linewidth=1)
            ax.axvline(0, color='black', linewidth=1)

            plt.show()

            return None

        else:
            raise TypeError('No se pueden graficar puntos ni funciones con mas de 3 dimensiones :p')

        plt.grid(True)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)


        plt.show()

class KNN():

    def __init__(self, K):

        self._estandarizado = False
        
        self.K = K
        

    
    

    def _estandarizar(self, datos: list[list]):

        datos = np.asarray(datos, dtype=object)

        x = np.asarray(datos[:, :-1], dtype=float)
        y = datos[:, -1]

        self.means = np.mean(x, axis=0)
        self.stds = np.std(x, axis=0)

        x_std = (x - self.means) / self.stds

        self._estandarizado = True
        return x_std, y



    def entrenar(self, datos: list[list], estandarizar=True):    

        if len(datos) == 0:
            raise IndexError('No hay datos')

        datos = np.asarray(datos, dtype=object)

        if estandarizar:
            self.x_train, self.y_train = self._estandarizar(datos)
        else:
            self.x_train = np.asarray(datos[:, :-1], dtype=float)
            self.y_train = np.asarray(datos[:, -1], dtype=int)

        self._entrenado = True
        return self



    def f_hat(self, x: list | np.ndarray, prob=False, beta=1.0):
        # 1. Calcular los términos al cuadrado
        # X_test_sq: shape (n_test, 1)
        if len(x.shape) != 2:
        
            # 2. Calcular el término cruzado (producto punto)
            # cross_term: shape (n_test, n_train)
            cross_term = np.dot(x, self.x_train.T)

            x_sq = np.sum(x**2)
            X_train_sq = np.sum(self.x_train**2, axis=1)
            
            # 3. Matriz de distancias euclidianas al cuadrado usando broadcasting
            # dists: shape (n_test, n_train)
            dists_sq = np.maximum(x_sq - 2 * cross_term + X_train_sq, 0.0)
            dists = np.sqrt(dists_sq)
            
            # 4. Obtener los índices de los K vecinos más cercanos para cada fila de prueba
            # k_indices: shape (n_test, k)
            k_indices = np.argsort(dists, axis=0)[:self.K]

            
            # 5. Mapear los índices a las etiquetas reales de y_train
            # k_labels: shape (n_test, k)
            k_labels = self.y_train[k_indices]

            # 6. Encontrar la etiqueta más común (moda) para cada muestra de prueba
            # Usamos un bucle por fila solo para la moda, o funciones vectorizadas si son clasificaciones binarias.
            # Para que sea 100% vectorizado en clasificación multi-clase genérica:

            predictions = np.bincount(k_labels).argmax()

        else:
                

            X_test_sq = np.sum(x**2, axis=1, keepdims=True)
            # X_train_sq: shape (1, n_train)
            X_train_sq = np.sum(self.x_train**2, axis=1, keepdims=False)
            
            # 2. Calcular el término cruzado (producto punto)
            # cross_term: shape (n_test, n_train)
            cross_term = np.dot(x, self.x_train.T)
            
            # 3. Matriz de distancias euclidianas al cuadrado usando broadcasting
            # dists: shape (n_test, n_train)
            dists = np.sqrt(X_test_sq - 2 * cross_term + X_train_sq)
            
            # 4. Obtener los índices de los K vecinos más cercanos para cada fila de prueba
            # k_indices: shape (n_test, k)
            k_indices = np.argsort(dists, axis=1)[:, :self.K]
            
            # 5. Mapear los índices a las etiquetas reales de y_train
            # k_labels: shape (n_test, k)
            k_labels = self.y_train[k_indices]
            
            # 6. Encontrar la etiqueta más común (moda) para cada muestra de prueba
            # Usamos un bucle por fila solo para la moda, o funciones vectorizadas si son clasificaciones binarias.
            # Para que sea 100% vectorizado en clasificación multi-clase genérica:

            predictions = np.array([np.bincount(row).argmax() for row in k_labels])



        if prob == True:

            logits_k = -dists / beta

            # 4. Acumular los logits por cada clase
            num_clases = len(np.unique(self.y_train))
            class_logits = np.zeros(num_clases, dtype=np.float32)
            
            for label, logit in zip(k_labels, logits_k):
                class_logits[int(label)] += logit

            # 5. Aplicar SOFTMAX Numéricamente Estable sobre la distribución de clases
            # Restamos el máximo logit para evitar overflow de e^x
            exp_logits = np.exp(class_logits - np.max(class_logits))
            probabilidades = exp_logits / np.sum(exp_logits)

            return predictions, probabilidades
        
        return predictions

    def f_hat_multiples_k(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, lista_ks: list[int]) -> dict[int, np.ndarray]:
        K_max = max(lista_ks)
        
        # 1. PASO PESADO: Calcular distancias una sola vez para todo el lote
        # (N_test x M_train)
        sum_X = np.sum(X_test ** 2, axis=1, keepdims=True)
        sum_train = np.sum(X_train ** 2, axis=1)
        distancias_sq = sum_X + sum_train - 2 * (X_test @ X_train.T)
        distancias_sq = np.maximum(distancias_sq, 0.0)

        # 2. Extraer y ordenar solo los K_max vecinos más cercanos (N_test x K_max)
        k_max_indices = np.argpartition(distancias_sq, K_max - 1, axis=1)[:, :K_max]
        
        rows = np.arange(X_test.shape[0])[:, None]
        dists_k = distancias_sq[rows, k_max_indices]
        orden_k = np.argsort(dists_k, axis=1)
        
        # Matriz con los K_max vecinos ordenados por cercanía (de menor a mayor distancia)
        indices_ordenados = k_max_indices[rows, orden_k]
        labels_k_max = y_train[indices_ordenados]  # Matriz de etiquetas (N_test x K_max)

        # 3. PASO RÁPIDO: Generar las predicciones para cada K haciendo 'slicing'
        resultados = {}
        
        for K in lista_ks:
            # Tomamos solo los primeros K vecinos de la matriz precalculada
            sub_labels = labels_k_max[:, :K]
            
            # Realizamos la votación para este K específico
            preds_k = np.zeros(X_test.shape[0], dtype=y_train.dtype)
            for idx, fila in enumerate(sub_labels):
                counts = np.bincount(fila)
                preds_k[idx] = np.argmax(counts)  # Moda / Voto mayoritario
                
            resultados[K] = preds_k

        return resultados


    def graficar_f(self, precision=100):

        if self._entrenado == False:
            raise AttributeError("No se ha entrenado el modelo")


        if len(self.x_train[0]) == 2:

            # ESTANDARIZAR CUANDO SE ESTANDARICE NO CUANDO NO BDFGD

            x_plot = self.x_train[:,0] * self.stds[0] + self.means[0]
            y_plot = self.x_train[:,1] * self.stds[1] + self.means[1]

            clases = np.unique(self.y_train)
            todos_los_colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

            enum = dict()
            colores = dict()

            for i in range(len(clases)):

                enum[clases[i]] = i
                colores[clases[i]] = todos_los_colores[i]



            x_min, x_max = np.min(x_plot) - 1, np.max(x_plot) + 1
            y_min, y_max = np.min(y_plot) - 1, np.max(y_plot) + 1
            
            xv,yv = np.meshgrid(np.linspace(x_min, x_max, precision), np.linspace(y_min, y_max, precision))

            par = np.column_stack((xv.ravel(), yv.ravel()))

            predicciones = np.array(
                [self.f_hat(fila) for fila in par],
                dtype=object
            )

            Z = np.array(
                [enum[clase] for clase in predicciones],
                dtype=int
            )
        
            Z = Z.reshape(xv.shape)


            plt.pcolormesh(xv, yv, Z)


            paint = [colores[self.y_train[i]] for i in range(len(x_plot))]


            if self._estandarizado == True:

                plt.scatter(x_plot, y_plot, c=paint)
            else:
                plt.scatter(x_plot, y_plot, c=paint)

        else:
            raise TypeError('No se pueden graficar fronteras de decisiones con 3 dimensiones (tal vez si pero no se veria bien) :p')



        plt.show()

        return self

class Regresion_logistica():


    def __init__(self, epochs: int, learning_rate: float, rand_params=True):

        self._estandarizado = False
        self._entrenado = False
        self._rand_params = rand_params
     
        self.epochs = epochs
        self.learning_rate = learning_rate

    
    

    def _estandarizar(self, datos: list[list]):

        x = datos[:, :-1]

        res = np.zeros_like(x)

        self.means = np.mean(x, axis=0)
        self.stds = np.std(x, axis=0)

        for i,d in enumerate(x):

            res[i] = [ (d[j] - self.means[j]) / self.stds[j] for j in range(len(d)) ]

        self._estandarizado = True

        return np.column_stack((res, datos[:, -1]))

    @staticmethod
    def _sigmoide(x):

        return 1 / (1 + np.exp(-x))


    def loss_func(self,params, datos):

        error = 0

        for i,d in enumerate(datos):

            error += d[-1] * np.log(self._sigmoide(np.dot(params[:-1], d[:-1]) + params[-1]))  +  (1 - d[-1]) * np.log(1 -  self._sigmoide(np.dot(params[:-1], d[:-1]) + params[-1]))

        return error


    
    def _gradiente(self, params, datos):

        res = np.zeros_like(params, dtype=float)


        # Derivada del intercepto b
        for i,d in enumerate(datos):

            res[-1] += (self._sigmoide(np.dot(params[:-1], d[:-1]) + params[-1]) - d[-1])


        # Derivadas de los pesos W
        for i,d in enumerate(datos):

            for j in range(len(res) - 1):

                res[j] += ( self._sigmoide( np.dot(params[:-1], d[:-1]) + params[-1] ) - d[-1] ) * d[j]

        return res / len(datos)


    def entrenar(self, datos: list[list], estandarizar=True):    

        if estandarizar == True:

            datos = self._estandarizar(datos=datos)

        if self._rand_params == True:
            parametros = np.random.uniform(-2, 2, size=(len(datos[0, :-1]) + 1 ))
        else:
            parametros = np.ones(shape=(len(datos[0, :-1]) + 1 ))

        
        for i in range(self.epochs):

            
            grad = self._gradiente(datos=datos, params=parametros)

            parametros += -self.learning_rate * grad


        self._params = parametros


        self._entrenado = True

        return self



    def f_hat(self, x: list):

        if self._entrenado == False:
            raise RuntimeError('No se ha entrenado el modelo :v')


        if self._estandarizado == True:

            x_std = (x - self.means) / self.stds

            x1 = np.concatenate((x_std, [1.0]))

            res = np.dot(self._params, x1)

            return self._sigmoide(res) 


        x1 = np.concatenate((x, [1.0]))
        
        res = np.dot( self._params, x1)

            

        return self._sigmoide(res)



    def MSE(self, y_pred, y):

        error = 0

        # for i in self.datos:

        #     equises = [i[j] for j in range(len(i) - 1)]


        #     error += (i[-1] - self.f_hat( equises ) )  *  (i[-1] - self.f_hat( equises ))

        for i in range(len(y_pred)):

            error += (y_pred[i] - y[i]) * (y_pred[i] - y[i])


        return error / len(y_pred)

    def R_2(self, y_pred, y):


        total_var = 0
        res_var = 0

        mean_total_var = np.mean(y)

        # for i in range(len(self.datos)):

        #     equises = [self.datos[i][j] for j in range(len(self.datos[i]) - 1)]

        #     res_var += (self.y[i] - self.f_hat( equises ) ) * (self.y[i] - self.f_hat( equises ))

        #     total_var += (self.y[i] - mean_total_var) * (self.y[i] - mean_total_var)

        for i in range(len(y_pred)):

            res_var += (y_pred[i] - y[i]) * (y_pred[i] - y[i])

            total_var += (y[i] - mean_total_var) * (y[i] - mean_total_var)


        return 1 - (res_var/total_var)


    def graficar_f(self,datos:list[list], frontera=False, puntos=True, final_percent=0.2, precision=100):

        if len(datos[0]) == 2:

            x_data = [i[0] for i in datos]
            y = [i[1] for i in datos]

            xmax = np.max(x_data)
            
            x_plot = np.linspace(0, round(xmax * (1 + final_percent)), precision)

            x_plot.resize((len(x_plot), len(datos[0]) - 1))

            y_plot = np.array([self.f_hat(x_plot[i]) for i in range(len(x_plot))])

            
            plt.plot(x_plot, y_plot)
            
            if puntos == False:
                plt.plot(x_data, y)
            else:
                plt.scatter(x_data, y)

            plt.grid(True)
            plt.axhline(0, color='black', linewidth=1)
            plt.axvline(0, color='black', linewidth=1)

            plt.show()

            return self


        elif len(datos[0]) == 3:

            if frontera == True:

                # if self._estandarizado == False:
                #     x_plot = datos[:,0] * self.stds[0] + self.means[0]
                #     y_plot = datos[:,1] * self.stds[1] + self.means[1]
                # else:
                #     x_plot = datos[:,0]
                #     y_plot = datos[:,1]

                import matplotlib.animation as anim

                fig, ax = plt.subplots()

                x_plot = datos[:,0]
                y_plot = datos[:,1]


                clases = np.unique(datos[:, -1])
                todos_los_colores = ['red', 'blue']

                colores = dict()

                for i in range(len(clases)):

                    colores[clases[i]] = todos_los_colores[i]



                x_min, x_max = np.min(x_plot) - 1, np.max(x_plot) + 1
                y_min, y_max = np.min(y_plot) - 1, np.max(y_plot) + 1
                
                xv,yv = np.meshgrid(np.linspace(x_min, x_max, precision), np.linspace(y_min, y_max, precision))

                par = np.column_stack((xv.ravel(), yv.ravel()))

                predicciones = np.array(
                    [self.f_hat(fila) for fila in par],
                    dtype=float
                ).reshape(xv.shape)

                Z = (predicciones >= 0.5).astype(int)



                ax.contour(
                    xv,
                    yv,
                    Z,
                    levels=[0.5],
                    colors='black',
                    linewidths=2
                )

                ax.pcolormesh(xv, yv, Z, cmap='coolwarm')


                paint = [colores[datos[:, -1][i]] for i in range(len(x_plot))]


                ax.scatter(x_plot, y_plot, c=paint)

                plt.show()

                return self




            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            x_data = [i[0] for i in datos]
            z_data = [i[1] for i in datos]


            xmax = np.max(x_data)
            ymax = np.max(z_data)
            
            x_plot = np.linspace(0, round(xmax * (1 + final_percent)), precision)

            
            x_plot = x_plot[:, np.newaxis]


            y_plot = np.linspace(0, round(ymax * (1 + final_percent)), precision)

            y_plot = y_plot[:, np.newaxis]


            xy_plot = np.column_stack((x_plot, y_plot))


            z_plot = np.array([self.f_hat(xy_plot[i]) for i in range(len(x_plot))])

            x_vals = np.linspace(min(x_data), max(x_data), 30)
            z_vals = np.linspace(min(z_data), max(z_data), 30)

            X, Z = np.meshgrid(x_vals, z_vals)

            Y = np.array([[self.f_hat([x, z]) for x, z in zip(row_x, row_z)]
                        for row_x, row_z in zip(X, Z)])         

            ax.plot_surface(X, Z, Y, alpha=0.7)
            
            
            if puntos == False:
                ax.plot(x_data, z_data, datos[:, -1])
            else:
                ax.scatter(x_data, z_data, datos[:, -1])

            ax.grid(True)

            plt.show()

            return self

        else:
            raise TypeError('No se pueden graficar puntos ni funciones con mas de 3 dimensiones :p')

class Regresion_Softmax():

    def __init__(self, epochs: int, learning_rate: float, architecture=(1,1), mini_batch=(True, 32), rand_params=True):

        self._estandarizado = False
        self._entrenado = False
        self._rand_params = rand_params

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.architecture = architecture
        self.mini_batch = mini_batch


     
        if self._rand_params == True:
            self._pesos = np.random.uniform(-2, 2, size=( architecture[0], architecture[-1] ))
            self._sesgos = np.random.uniform(-2, 2, size=architecture[-1])
        else:
            self._pesos = np.ones(shape=( architecture[0], architecture[-1] ))
            self._sesgos = np.ones(shape=architecture[-1])         

        # if self._rand_params == True:
        #     self._pesos = np.random.uniform(-2, 2, size=( len(datos[0]) - 1, architecture[-1] ))
        #     self._sesgos = np.random.uniform(-2, 2, size=architecture[-1])
        # else:
        #     self._pesos = np.ones(shape=( len(datos[0]) - 1, len(clases) ))
        #     self._sesgos = np.ones(shape=len(clases))         

    
    

    def _estandarizar(self, datos: list[list]):

        x = datos[:, :-1]


        res = np.zeros_like(x)

        self.means = np.mean(x, axis=0)
        self.stds = np.std(x, axis=0)

        for i,d in enumerate(x):

            res[i] = [ (d[j] - self.means[j]) / self.stds[j] for j in range(len(d)) ]

        self._estandarizado = True

        return np.column_stack((res, datos[:, -1]))


    def _softmax(self, x: np.ndarray):

        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        
        # Dividir por la suma de las exponenciales
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def loss_func(self,params, datos):

        error = 0

        for i,d in enumerate(datos):

            error += d[-1] * np.log(self._softmax(np.dot(params[:-1], d[:-1]) + params[-1]))  +  (1 - d[-1]) * np.log(1 -  self._softmax(np.dot(params[:-1], d[:-1]) + params[-1]))

        return error

    def crear_minibatches(self, X, y):
        assert len(X) == len(y), "X e y deben tener el mismo tamaño"
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
            
        for i in range(0, len(X), self.mini_batch[1]):
            X_batch = X[i:i + self.mini_batch[1]]
            y_batch = y[i:i + self.mini_batch[1]]
            yield X_batch, y_batch

    
    def _gradiente(self,caract: list[list], targets: list[list]) -> tuple:


        z = caract @ self._pesos + self._sesgos

        proba = np.apply_along_axis(self._softmax, axis=1, arr=z)

        error = proba - targets


        w_grad = np.dot(caract.T, error) / len(caract)

        b_grad = np.sum(error, axis=0)


        return (w_grad, b_grad)


    def entrenar(self, datos: list[list], estandarizar=True):    

        if estandarizar == True:

            datos = self._estandarizar(datos=datos)      


        objetivos = datos[:, -1]

        clases = np.unique(objetivos)

        caract = datos[:, :-1]


        targets = np.array([(n == clases).astype(dtype=int) for i,n in enumerate(objetivos)])

        error = []
        
        for i in range(self.epochs):

            if self.mini_batch[0] == True:


                for X,y in self.crear_minibatches(X=caract, y=targets):

                    grad = self._gradiente(caract=X, targets=y)


                    self._pesos -= self.learning_rate * grad[0]
                    self._sesgos -= self.learning_rate * grad[1]

                
            else:

                grad = self._gradiente(caract=caract, targets=targets)

                self._pesos -= self.learning_rate * grad[0]
                self._sesgos -= self.learning_rate * grad[1]
    


        self._entrenado = True

        return self



    def f_hat(self, x: list):



        if self._estandarizado == True:

            x_std = (x - self.means) / self.stds

            x1 = np.array(x_std)

            res = self._softmax(  x1 @ self._pesos + self._sesgos  )

            return res


        x1 = np.array(x)
        
        res = self._softmax(  x1 @ self._pesos + self._sesgos  )

        return res



    def MSE(self, y_pred, y):

        error = 0

        for i in range(len(y_pred)):

            error += (y_pred[i] - y[i]) * (y_pred[i] - y[i])


        return error / len(y_pred)

    def R_2(self, y_pred, y):


        total_var = 0
        res_var = 0

        mean_total_var = np.mean(y)

        # for i in range(len(self.datos)):

        #     equises = [self.datos[i][j] for j in range(len(self.datos[i]) - 1)]

        #     res_var += (self.y[i] - self.f_hat( equises ) ) * (self.y[i] - self.f_hat( equises ))

        #     total_var += (self.y[i] - mean_total_var) * (self.y[i] - mean_total_var)

        for i in range(len(y_pred)):

            res_var += (y_pred[i] - y[i]) * (y_pred[i] - y[i])

            total_var += (y[i] - mean_total_var) * (y[i] - mean_total_var)


        return 1 - (res_var/total_var)


    def graficar_f(self,datos:list[list], frontera=False, puntos=True, final_percent=0.2, precision=100):

        if len(datos[0]) == 2:

            x_data = [i[0] for i in datos]
            y = [i[1] for i in datos]

            xmax = np.max(x_data)
            
            x_plot = np.linspace(0, round(xmax * (1 + final_percent)), precision)

            x_plot.resize((len(x_plot), len(datos[0]) - 1))

            y_plot = np.array([self.f_hat(x_plot[i]) for i in range(len(x_plot))])

            
            plt.plot(x_plot, y_plot)
            
            if puntos == False:
                plt.plot(x_data, y)
            else:
                plt.scatter(x_data, y)

            plt.grid(True)
            plt.axhline(0, color='black', linewidth=1)
            plt.axvline(0, color='black', linewidth=1)

            plt.show()

            return self


        elif len(datos[0]) == 3:

            if frontera == True:


                fig, ax = plt.subplots()

                x_plot = datos[:,0]
                y_plot = datos[:,1]


                clases = np.unique(datos[:, -1])
                todos_los_colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

                colores = dict()

                for i in range(len(clases)):

                    colores[clases[i]] = todos_los_colores[i]



                x_min, x_max = np.min(x_plot) - 1, np.max(x_plot) + 1
                y_min, y_max = np.min(y_plot) - 1, np.max(y_plot) + 1
                
                xv,yv = np.meshgrid(np.linspace(x_min, x_max, precision), np.linspace(y_min, y_max, precision))

                par = np.column_stack((xv.ravel(), yv.ravel()))


                predicciones = np.array(
                    [self.f_hat(fila) for fila in par],
                    dtype=float
                ).reshape(xv.shape)


                Z = predicciones



                ax.contour(
                    xv,
                    yv,
                    Z,
                    levels=np.arange(0.5, len(clases) - 0.5 + 1, 1),
                    colors="black"
                )

                ax.pcolormesh(xv, yv, Z, cmap='coolwarm')


                paint = [colores[datos[:, -1][i]] for i in range(len(x_plot))]


                ax.scatter(x_plot, y_plot, c=paint)

                plt.show()

                return self




            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            x_data = [i[0] for i in datos]
            z_data = [i[1] for i in datos]


            xmax = np.max(x_data)
            ymax = np.max(z_data)
            
            x_plot = np.linspace(0, round(xmax * (1 + final_percent)), precision)

            
            x_plot = x_plot[:, np.newaxis]


            y_plot = np.linspace(0, round(ymax * (1 + final_percent)), precision)

            y_plot = y_plot[:, np.newaxis]


            xy_plot = np.column_stack((x_plot, y_plot))


            # z_plot = np.array([self.f_hat(xy_plot[i]) for i in range(len(x_plot))])

            x_vals = np.linspace(min(x_data), max(x_data), 30)
            z_vals = np.linspace(min(z_data), max(z_data), 30)

            X, Z = np.meshgrid(x_vals, z_vals)

            Y = np.array([[self.f_hat([x, z]) for x, z in zip(row_x, row_z)]
                        for row_x, row_z in zip(X, Z)])         

            ax.plot_surface(X, Z, Y, alpha=0.7)
            
            
            if puntos == False:
                ax.plot(x_data, z_data, datos[:, -1])
            else:
                ax.scatter(x_data, z_data, datos[:, -1])

            ax.grid(True)

            plt.show()

            return self

        else:
            raise TypeError('No se pueden graficar puntos ni funciones con mas de 3 dimensiones :p')    


def lector(x:list):

    for i,n in enumerate(x):

        print(n)
        print()


class MLP():

    def __init__(self, epochs: int, learning_rate: float, architecture: tuple, loss: Literal['mse', 'log'], activation: Literal['relu', 'sigmoid', 'tanh'] = 'sigmoid',
                optimizer: Literal['batch'] | tuple[Literal['minibatch'], int] = 'batch', rand_params=True):

        self._estandarizado = False
        self._entrenado = False
        self._rand_params = rand_params
        self.architecture = architecture
        self.optimizer = optimizer

        self.epochs = epochs
        self.learning_rate = learning_rate


        if activation == 'relu':

            self._activation = lambda x: np.where(x > 0, x, 0.01*x)

            self._der_activation = lambda x: np.where(x > 0, 1, 0.01)

        elif activation == 'sigmoid':

            self._activation = lambda x: 1/(1 + np.exp(-x))

            self._der_activation = lambda x: self._activation(x) * (1 - self._activation(x))


        self.loss = loss
        if loss == 'mse':

            self._der_loss = lambda a,y: a - y

        elif loss == 'log':

            self._der_loss = lambda a,y: a - y

        rng = np.random.default_rng(6767)

        if rand_params == True:
            self._pesos = [rng.standard_normal(size=(architecture[i], architecture[i + 1])) for i in range(len(architecture) - 1)]  
        else:
            self._pesos = [np.ones(shape=(architecture[i], architecture[i + 1])) for i in range(len(architecture) - 1)]  


        self._sesgos = [np.zeros(shape=architecture[i]) for i in range(1, len(architecture))]

    
    

    def _estandarizar(self, datos: list[list]):

        x = datos[:, :-1]


        res = np.zeros_like(x)

        self.means = np.mean(x, axis=0)
        self.stds = np.std(x, axis=0)

        for i,d in enumerate(x):

            res[i] = [ (d[j] - self.means[j]) / self.stds[j] for j in range(len(d)) ]

        self._estandarizado = True

        return np.column_stack((res, datos[:, -1]))


    def _activation(self, x: float):

        pass

    def _der_activation(self, x: float):

        pass

    def _der_loss(self, a: np.ndarray, y: np.ndarray):

        pass

    def _softmax(self, x: np.ndarray):

        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        
        # Dividir por la suma de las exponenciales
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def crear_minibatches(self, X, y):
        assert len(X) == len(y), "X e y deben tener el mismo tamaño"
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
            
        for i in range(0, len(X), self.optimizer[1]):
            X_batch = X[i:i + self.optimizer[1]]
            y_batch = y[i:i + self.optimizer[1]]
            yield X_batch, y_batch

    
    def _backward_pass(self, x, y, categoric) -> tuple:

        dw = [np.zeros_like(i) for i in self._pesos]
        db = [np.zeros_like(i) for i in self._sesgos]

        z = [np.zeros(shape=(len(y), len(s))) for i,s in enumerate(self._sesgos)]
        a = [x, *[m.copy() for m in z]]

        deltas = [m.copy() for m in z]


        # Calcula de Zs y As de cada capa (sin incluir la entrada x)
        for i in range(len(self.architecture) - 1):
            # print(f"{a[i]} @ {self._pesos[i]} + {self._sesgos[i]}")
            # input()

            res = a[i] @ self._pesos[i] + self._sesgos[i]

            # print(f"{res}")
            # input()

            z[i] = res

            # Se suma 1 porque el primer elemento es x y x no se cambia
            if i < len(self.architecture) - 2:
                
                a[i + 1] = self._activation(res)
                continue

            # print('siuu')
            # input()

            a[i + 1] = res


        # lector(z)
        # print()
        # lector(a)
        # input()




        # Calcula de deltas de la capa de salida
        deltas[-1] = self._der_loss(a=a[-1] if categoric == False else self._softmax(a[-1]), y=y)

        # lector(deltas[-1])
        # input()
            

        # Calculo de deltas de cada capa
        for i in range(len(z) - 2, -1, -1):

            # if i < len(z) - 3:
            #     deltas[i] = (
            #         deltas[i + 1] @ self._pesos[i + 1].T
            #     ) * self._der_activation(z[i])
            #     continue

            deltas[i] = (
                deltas[i + 1] @ self._pesos[i + 1].T
            ) * self._der_activation(z[i])


        # Calculo de gradientes de cada parametro usando los deltas
        for i in range(len(deltas)):

            dw[i] = np.dot(a[i].T, deltas[i]) / len(y)
            db[i] = np.sum(deltas[i], axis=0) / len(y)


        # for ind,ex in enumerate(x):

        #     z = [ex]
        #     a = [ex]

        #     deltas = [np.zeros_like(i) for i in self._sesgos]


        #     for i in range(len(self.architecture) - 1):

        #         # print(f"{a[-1]} @ {self._pesos[i]} + {self._sesgos[i]}")
        #         res = a[-1] @ self._pesos[i] + self._sesgos[i]
        #         # print(f"{res}")
        #         # input()                    
        #         z.append(res)

        #         if i < len(self.architecture) - 2:

        #             a.append(self._activation(res)) 

        #             continue

        #         a.append(res)

        #     # lector(z)
        #     # print()
        #     # lector(a)
        #     # input()


        #     deltas[-1] = self._der_loss(a[-1] if categoric == False else self._softmax(a[-1]) , y[ind])

        #     dw[-1] += np.outer(a[-2], deltas[-1])

        #     db[-1] += deltas[-1]



        #     for i in range(1, len(self.architecture) - 1):   

        #         # -1 - i como indice ya que asi vamos de adelante hacia atras
        #         for j,zj in enumerate(z[-1 - i]):

        #             # CALCULO DE DELTAS
        #             deltas[-1 - i][j] = self._der_activation(zj)

        #             # print(f"{deltas[-1 - i + 1]}  *  {self._pesos[-1 -i + 1][j]}")
        #             # input()

        #             mult = np.dot(deltas[-1 - i + 1], self._pesos[-1 - i + 1][j])


        #             # print(f"{deltas[-1 - i + 1]}  *  {self._pesos[-1 - i][j]}")

        #             deltas[-1 - i][j] *= mult 



        #         dw[-1 -i] += np.outer(a[-1 -i - 1],deltas[-1 -i])
        #         db[-1 -i] += deltas[-1 -i]


        # for i,d in enumerate(db):

        #     dw[i] /= len(y)
        #     db[i] /= len(y)


        return (dw, db)
    


    def entrenar(self, datos: list[list], estandarizar=True):    

        if estandarizar == True:

            datos = self._estandarizar(datos=datos)


        x = 0
        y = 0
        categoric = False
        codif = None

        if self.loss == "log":
            targets = np.unique(datos[:, -1])

        len_out = -self.architecture[-1]

        if self.loss == 'mse':

            if len(datos[0][:, :len_out]) != self.architecture[0]:

                raise RuntimeError(f"Hay {len(datos[0][:, :len_out])} entrada/s y {self.architecture[0]} entrada/s en la red neuronal no son compatibles xd")

            def codificar(data) -> tuple:

                x = data[:, :len_out]

                y = data[:, len_out:]

                return (x,y)

            codif = codificar

        elif self.loss == 'log':

            if len(datos[0][:-1]) != self.architecture[0]:

                raise RuntimeError(f"Hay {len(datos[0][:-1])} entrada/s y {self.architecture[0]} entrada/s en la red neuronal no son compatibles xd")

            objs = np.unique(datos[:, -1])

            if len(objs) != self.architecture[-1]:

                raise RuntimeError(f"La cantidad de clases ({len(objs)}) no es la misma que la cantidad de salidas ({self.architecture[-1]}) lol")

            categoric = True

            def codificar(data) -> tuple:
                x = data[:, :-1]

                ys = data[:, -1]

                y = np.asarray([(targets == i).astype(dtype=int) for i in ys])

                return x, y

            codif = codificar


        rng = np.random.default_rng(6767)

        if self.optimizer == 'batch':
            x,y = codif(datos)

            for i in range(self.epochs):

                grad = self._backward_pass(x=x, y=y, categoric=categoric)
                
                for i in range(len(self.architecture) - 1):

                    self._pesos[i] -= self.learning_rate * grad[0][i]
                    self._sesgos[i] -= self.learning_rate * grad[1][i]

            self._entrenado = True

            return self

        caract,ys = codif(datos)

        for i in range(self.epochs):


            for x,y in self.crear_minibatches(X=caract,y=ys):

                grad = self._backward_pass(x=x, y=y, categoric=categoric)
                
                for i in range(len(self.architecture) - 1):

                    self._pesos[i] -= self.learning_rate * grad[0][i]
                    self._sesgos[i] -= self.learning_rate * grad[1][i]




        self._entrenado = True

        return self



    def f_hat(self, x: list):

        # if len(x[0]) != self.architecture[0]:

        #     raise RuntimeError(f"Hay {len(x)} entrada/s y {self.architecture[0]} entrada/s en la red neuronal no son compatibles xd")

        if self._estandarizado == True:

            x_std = (x - self.means) / self.stds

            x1 = np.asarray(x_std)

            res = x1

            for i in range(len(self.architecture) - 2):

                res = self._activation( res @ self._pesos[i] + self._sesgos[i] )


            return self._softmax(res @ self._pesos[-1] + self._sesgos[-1])


        x1 = np.asarray(x)

        res = x1

        
        for i in range(len(self.architecture) - 2):

            res = self._activation( (res @ self._pesos[i]) + self._sesgos[i] )

        
        return self._softmax(res @ self._pesos[-1] + self._sesgos[-1])



    def MSE(self, y_pred, y):

        error = 0

        for i in range(len(y_pred)):

            error += (y_pred[i] - y[i]) * (y_pred[i] - y[i])


        return error / len(y_pred)

    def R_2(self, y_pred, y):


        total_var = 0
        res_var = 0

        mean_total_var = np.mean(y)

        # for i in range(len(self.datos)):

        #     equises = [self.datos[i][j] for j in range(len(self.datos[i]) - 1)]

        #     res_var += (self.y[i] - self.f_hat( equises ) ) * (self.y[i] - self.f_hat( equises ))

        #     total_var += (self.y[i] - mean_total_var) * (self.y[i] - mean_total_var)

        for i in range(len(y_pred)):

            res_var += (y_pred[i] - y[i]) * (y_pred[i] - y[i])

            total_var += (y[i] - mean_total_var) * (y[i] - mean_total_var)


        return 1 - (res_var/total_var)


    def graficar_f(self,datos:list[list], frontera=False, puntos=True, final_percent=0.2, precision=100):

        if len(datos[0]) == 2:

            x_data = [i[0] for i in datos]
            y = [i[1] for i in datos]

            xmax = np.max(x_data)
            
            x_plot = np.linspace(0, round(xmax * (1 + final_percent)), precision)

            x_plot.resize((len(x_plot), len(datos[0]) - 1))

            y_plot = np.array([self.f_hat(x_plot[i]) for i in range(len(x_plot))])

            
            plt.plot(x_plot, y_plot)
            
            if puntos == False:
                plt.plot(x_data, y)
            else:
                plt.scatter(x_data, y)

            plt.grid(True)
            plt.axhline(0, color='black', linewidth=1)
            plt.axvline(0, color='black', linewidth=1)

            plt.show()

            return self


        elif len(datos[0]) == 3:

            if frontera == True:


                fig, ax = plt.subplots()

                x_plot = datos[:,0]
                y_plot = datos[:,1]


                clases = np.unique(datos[:, -1])
                todos_los_colores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                todos_los_colores_light = ['lightcoral', 'lightblue', 'lightgreen', 'peachpuff', 'plum', 'rosybrown', 'lavenderblush', 'lightgray']

                colores = dict()

                for i in range(len(clases)):

                    colores[clases[i]] = todos_los_colores[i]


                x_min, x_max = np.min(x_plot) - 1, np.max(x_plot) + 1
                y_min, y_max = np.min(y_plot) - 1, np.max(y_plot) + 1
                
                xv,yv = np.meshgrid(np.linspace(x_min, x_max, precision), np.linspace(y_min, y_max, precision))

                par = np.column_stack((xv.ravel(), yv.ravel()))

                predicciones = np.array(
                    [np.argmax(self.f_hat(fila)) for fila in par],
                    dtype=float
                ).reshape(xv.shape)



                Z = predicciones


                ax.contour(
                    xv,
                    yv,
                    Z,
                    levels=np.arange(0.5, len(clases) - 0.5 + 1, 1),
                    colors="black"
                )

                cmap = ListedColormap(todos_los_colores_light[:len(clases)])

                ax.pcolormesh(xv, yv, Z, cmap=cmap)

                paint = [colores[datos[:, -1][i]] for i in range(len(x_plot))]


                ax.scatter(x_plot, y_plot, c=paint)

                plt.show()

                return self




            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

            x_data = [i[0] for i in datos]
            z_data = [i[1] for i in datos]


            xmax = np.max(x_data)
            ymax = np.max(z_data)
            
            x_plot = np.linspace(0, round(xmax * (1 + final_percent)), precision)

            
            x_plot = x_plot[:, np.newaxis]


            y_plot = np.linspace(0, round(ymax * (1 + final_percent)), precision)

            y_plot = y_plot[:, np.newaxis]


            xy_plot = np.column_stack((x_plot, y_plot))



            x_vals = np.linspace(min(x_data), max(x_data), 30)
            z_vals = np.linspace(min(z_data), max(z_data), 30)

            X, Z = np.meshgrid(x_vals, z_vals)

            Y = np.array([[self.f_hat([x, z]) for x, z in zip(row_x, row_z)]
                        for row_x, row_z in zip(X, Z)])         

            ax.plot_surface(X, Z, Y, alpha=0.7)
            
            
            if puntos == False:
                ax.plot(x_data, z_data, datos[:, -1])
            else:
                ax.scatter(x_data, z_data, datos[:, -1])

            ax.grid(True)

            plt.show()

            return self

        else:
            raise TypeError('No se pueden graficar puntos ni funciones con mas de 3 dimensiones :p')    


def val_cruz(modelo: Modelo_base, datos: list[list], transform_class, medida: Literal['regresion','clasificacion'], n_splits=4, shuffle=True):



    # Resto de la division
    num_rest = len(datos) % n_splits
    # Se le resta el resto a la cantidad de datos para que sea entero
    num_folds = (len(datos) - num_rest)//n_splits
    # Array de los datos divido en K carpetas
    k_folds = []
    # Desordenar los datos
    if shuffle == True:
        np.random.default_rng(seed=777).shuffle(x=np.asarray(datos, dtype=object))


    k_folds = np.array_split(np.asarray(datos, dtype=object), n_splits)



    # Bucle de shuffle para entrenar y testear

    actual = 0

    total_mse = 0

    acertadas = 0
    equivocadas = 0


    for i in range(n_splits):


        datos_train = None
        datos_test = None

        for i,fold in enumerate(k_folds):


            if i == actual:

                datos_test = fold.astype(dtype=np.float64)
                continue

            datos_train = fold.astype(dtype=np.float64)
                    

        # FASE DE TESTEO PARA REGRESION Y CLASIFICACION

        modelo.entrenar(datos=datos_train)

        if medida == 'regresion':

            total_mse += modelo.MSE()

            continue


        for i in datos_test:

            predict = transform_class(modelo.f_hat(i[:-1]))

            # print(f"pred: {predict} ;  true: {i[-1]}")
            # print(predict == i[-1])
            # input()

            if predict == i[-1]:
                acertadas += 1
                continue 

            equivocadas += 1

        actual += 1

    if medida == 'regresion':

        return total_mse/n_splits

    return (acertadas, equivocadas)

class StratifiedKFold:
    def __init__(self, n_splits=4, shuffle=True, seed=777, batch_size=500):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size  # Lote máximo para evitar saturar la RAM

    def split_indices(self, y: np.ndarray):
        """Genera solo los ÍNDICES de cada fold para no duplicar matrices en RAM."""
        clases, counts = np.unique(y, return_counts=True)
        folds_indices = [[] for _ in range(self.n_splits)]
        rng = np.random.default_rng(seed=self.seed)

        for cls in clases:
            cls_indices = np.where(y == cls)[0]
            if self.shuffle:
                rng.shuffle(cls_indices)
            
            cls_splits = np.array_split(cls_indices, self.n_splits)
            for fold_idx in range(self.n_splits):
                folds_indices[fold_idx].extend(cls_splits[fold_idx])

        folds_indices = [np.array(f, dtype=np.int32) for f in folds_indices]

        for test_fold_idx in range(self.n_splits):
            test_idx = folds_indices[test_fold_idx]
            train_idx = np.hstack([folds_indices[i] for i in range(self.n_splits) if i != test_fold_idx])
            yield train_idx, test_idx

    def test(self, modelo, X: np.ndarray, y: np.ndarray, transform_class=None) -> tuple[int, int]:
        # Convertir a float32 o uint8 para consumir la mitad/cuarta parte de RAM
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        acertadas = 0
        equivocadas = 0

        for train_idx, test_idx in self.split_indices(y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # Reestructurar datos para entrenamiento sin hacer copias pesadas
            datos_train = np.column_stack((X_train, y_train))
            modelo.entrenar(datos=datos_train, estandarizar=False)

            # -----------------------------------------------------------------
            # PREDICCIÓN POR MINILOTES (Previene el congelamiento del PC)
            # -----------------------------------------------------------------
            num_test = len(X_test)
            preds_list = []

            for start_idx in range(0, num_test, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_test)
                X_batch = X_test[start_idx:end_idx]

                # Predicción en un lote pequeño (p. ej. 500 muestras a la vez)
                batch_preds = modelo.f_hat(X_batch)
                if transform_class is not None:
                    batch_preds = transform_class(batch_preds)
                
                preds_list.append(batch_preds)

            # Concatenar solo las predicciones finales (vectores 1D ligeros)
            preds = np.concatenate(preds_list)

            # Evaluación de aciertos
            correctas = np.sum(preds == y_test)
            acertadas += correctas
            equivocadas += (len(y_test) - correctas)

        return int(acertadas), int(equivocadas)


def strat_val(modelo: Modelo_base, datos: list[list], transform_class, medida: Literal['regresion','clasificacion'] ,n_splits=4, shuffle=True):

    datos = np.asarray(datos, dtype=object)


    if shuffle == True:
        np.random.default_rng(seed=777).shuffle(x=np.asarray(datos, dtype=object))


    clases = np.unique(datos[:,-1])
    # Array de los datos divido en K carpetas
    k_folds = np.array_split(datos[datos[:, -1] == clases[0]], n_splits)

    
    for j,clase in enumerate(clases[1:]):

        division = np.array_split(datos[datos[:, -1] == clase], n_splits)

        for split in range(n_splits):

            k_folds[split] = np.vstack((k_folds[split], division[split]))



    # Bucle de shuffle para entrenar y testear

    actual = 0

    total_mse = 0

    acertadas = 0
    equivocadas = 0


    for i in range(n_splits):

        datos_train = None
        datos_test = None

        for i,fold in enumerate(k_folds):


            if i == actual:

                datos_test = fold.astype(dtype=np.float64)
                continue

            datos_train = fold.astype(dtype=np.float64)
                    

        # FASE DE TESTEO PARA REGRESION Y CLASIFICACION

        modelo.entrenar(datos=datos_train, estandarizar=False)

        if medida == 'regresion':

            total_mse += modelo.MSE()

            continue

        print('prediciendo')
        # SI NO FUNCIONA ES PORQUE F_HAT NO ESTA VECTORIZADA
        preds = transform_class(modelo.f_hat(datos_test[:, :-1]))
        print('terminado')
            

        for i,p in enumerate(preds):


            if p == datos_test[i][-1]:
                acertadas += 1
                continue

            equivocadas += 1



        actual += 1

    if medida == 'regresion':

        return total_mse/n_splits

    return (acertadas, equivocadas)




if __name__ == "__main__":

    # csv = pd.read_csv(r"C:\Users\USUARIO\Desktop\Proyectos\Python\archivos\Iris.csv")

    # datos = csv.iloc[:51, [3,4,5]]


    # datos.loc[datos['Species'] == 'Iris-setosa', 'Species'] = 0
    # datos.loc[datos['Species'] == 'Iris-versicolor', 'Species'] = 1
    # datos.loc[datos['Species'] == 'Iris-virginica', 'Species'] = 2


    # datos = datos.to_numpy(dtype=float)

    # # datos = np.array([
    # #     [1,2,0],
    # #     [2,3,1],
    # #     [3,0,1],
    # #     [4,1,0]
    # # ])

    # from time import perf_counter

    # modelo = MLP(epochs=10000, learning_rate=0.1, activation='relu', loss='log', rand_params=True, architecture=(2,1,2))

    # time = perf_counter()
    # modelo.entrenar(datos=datos, estandarizar=False)
    # print(f"tiempo: {(perf_counter() - time):.2f}s\n")

    # pred = ( datos[:,-1] == np.argmax(modelo.f_hat(datos[:,:-1]), axis=1) ).astype(dtype=int)

    # acertadas = np.sum(pred)
    # equivocadas = len(pred) - acertadas

    # print(f"acertados: {acertadas}")
    # print(f"equivocados: {equivocadas}")
    # print(f"precision: {(acertadas/len(pred) * 100):.2f}%")

    # modelo.graficar_f(datos=datos, frontera=True, precision=200)

    datos = np.array([
        [1,0],
        [2,1],
        [3,1],
        [5,2],
        [7,3]
    ])

    mod = KNN(K=5)

    mod.entrenar(datos=datos, estandarizar=False)

    print(mod.f_hat(x=np.array([2]), prob=True))

