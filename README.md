# 🧠 Proyecto: Clasificación de Obesidad con IA
Este proyecto tiene fines académicos y constituye una primera aproximación al uso de Inteligencia Artificial para clasificar el **índice de obesidad** de una persona, utilizando datos personales, hábitos y estilo de vida.  
Se ha desarrollado una aplicación web interactiva usando **Streamlit** que permite al usuario introducir sus datos y obtener una predicción automática del nivel de obesidad.

---

## 🚀 Demo del Proyecto
🔗 Accede a la demo:  
"https://proyectoobesidad-nodd3r-pabloescarda.streamlit.app"

> ⚠️ Recomendamos utilizar navegadores como **Opera**, **Opera GX** o **Google Chrome** para evitar posibles errores de compatibilidad con Streamlit.

---

La app 🖥️ permite al usuario introducir los siguientes datos para obtener una predicción:

    | Entrada | Tipo | Descripción |
    |--------|------|-------------|
    | Edad | Numérico | Edad del usuario |
    | Altura | Numérico (2 decimales) | Altura en metros |
    | Peso | Numérico (2 decimales) | Peso en kilogramos |
    | Frecuencia de consumo de alcohol | Desplegable | Opciones predefinidas |
    | Medio de transporte habitual | Desplegable | Opciones predefinidas |
    | Frecuencia de consumo de vegetales | Slider (1 a 3) | De baja a alta |
    | Número de comidas fuertes al día | Slider (1 a 6) | Número aproximado |
    | Actividad física semanal | Slider (0 a 3) | Nivel de actividad física |
    | ¿Consume alimentos hipercalóricos? | Checkbox | Sí/No |
    | ¿Es fumador? | Checkbox | Sí/No |
    | ¿Tiene antecedentes familiares de obesidad? | Checkbox | Sí/No |
    
    Por último hay un botón que al pulsar te da el resultado del test

    ![image](https://github.com/user-attachments/assets/0fbe7133-58a8-41c9-8928-7ae0149850ee)

---
    
## ⚙️ Instalación y Ejecución Local

1. Clona este repositorio:
   ```bash
   git clone https://github.com/usuario/proyecto_obesidad.git
   cd proyecto_obesidad
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecuta la app:
   ```bash
   streamlit run app.py
   ```
   
---

## 📌 Notas

- Este proyecto es únicamente para **fines educativos**.
- Se han usado herramientas como Google Colab para el entrenamiento y evaluación de los modelos.
- Si deseas replicar el entrenamiento, puedes revisar los notebooks en la carpeta correspondiente.

---

## 🙌 Autor
- Pablo Escarda

---

## ✅ Mejoras posibles
- Añadir validación cruzada y tuning de hiperparámetros.
- Añadir gráficos explicativos en la app (por ejemplo, `SHAP` o `Feature Importance`).
- Guardar logs del modelo y métricas.
- Añadir sección de interpretación del resultado para el usuario.
