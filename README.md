# README DE LA TAREA 1:

Tarea del Bloque 1: Introducción. Máster MIAX.

<p align="justify">
¿En qué consiste la tarea? La tarea consiste en crear un programa capaz de extraer datos históricos tanto de acciones como de índices (información bursátil), los cuales deben ser validados y limpiados para, posteriormente, mostrar información relevante sobre ellos, así como gráficos y una o varias simulaciones de Monte Carlo.
</p>

Cosas a destacar en este README: 
<p align="justify">
- Estructura: la estructura de mi práctica consiste en una serie de clases, sin herencias (no he sabido como implementarlas de manera coherente), las cuales tienen cada una un objetivo claro y distinto de las otras:
        - DataExtractor: descarga de datos históricos de varios orígenes (APIs). En esta clase he incluido una función que             descargue datos de criptomonedas de CoinGecko. Además, en esta clase he incluido todo el tema de normalización,              validación, limpieza y datos estadísticos.
        - PriceBar: representa una observación individual de precios.
        - PriceSeries: agrupa todas las barras de PriceBar (series completas de precios) de un solo activo/ticker.
        - Helpers: funciones auxiliares que no pertenecen a ninguna clase en específico. 
        - Portfolio: esta clase tiene como función principal la representación de la cartera/tickers. En ella se ha definido           la simulación de Monte Carlo y el report/plots_report.
</p>
- Carpeta src: 

<p align="justify">
- Como se ha realizado la práctica: de las tres preguntas que hay que responder en el vídeo sobre esta práctica ("qué", "comó" y "porqué"), en este apartado quiero aclarar el "como". El código de esta práctica ha sido creado mediante la ayuda de un LLM (GPT). Tanto la lógica detrás del programa como el resto de cosas que no son código son responsabilidad mía, pero es evidente que el código ha sido redactado mediante la ayuda de una IA. Hubo un primer intento de hacer la práctica sin ningún tipo de ayuda pero fue inviable: tanto la programación como las finanzas son unos campos prácticamente nuevos para mí. Pese a que creo que es evidente que para la realización de esta práctica debía hacerse uso de la IA o cualquier otro tipo de ayuda externa, sentia la necesidad de dejar claro en el README específicamente el "comó" porque es asi como se ha realizado la práctica, no hay pretensión de ocultar el procedimiento utilizado. 
</p>
