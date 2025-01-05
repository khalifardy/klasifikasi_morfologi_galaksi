import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, InceptionV3

def Resnet50():
    # Load ResNet-50 pre-trained tanpa fully connected layer (include_top=False)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Tambahkan lapisan fully connected untuk klasifikasi morfologi galaksi
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 3 kelas: elliptical, spiral, irregular

    # Buat model akhir
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    
    return model
    

def efficientnet80():
    # Load EfficientNetB0 pre-trained tanpa fully connected layer
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Tambahkan lapisan fully connected
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Buat model akhir
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    return model

def InterceptionV3():
    # Load InceptionV3 pre-trained tanpa fully connected layer
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 256, 3))

    # Tambahkan lapisan fully connected
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Buat model akhir
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    return model
