# -*- coding: utf-8 -*-
import sys
import os
import io
import weakref
import pandas as pd
import numpy as np
import re
import jieba
import random
import matplotlib
matplotlib.use('Qt5Agg')  # 设置matplotlib后端
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QThread 
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, Dense, SpatialDropout1D,
    BatchNormalization, Dropout, Activation, GlobalMaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from main_ui import Ui_Form as MainUI
from config_ui import Ui_Form as ConfigUI
from result_ui import Ui_Form as ResultUI
from log_ui import Ui_Form as LogUI

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# ====================== 预训练词向量加载 ======================
def load_pretrained_embeddings(file_path, tokenizer, embedding_dim):
    try:
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline().split()
            embedding_dim = int(header[1])
            for line in f:
                values = line.rstrip().split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        
        return embedding_matrix
    except Exception as e:
        print(f"加载预训练词向量失败: {str(e)}")
        return None

# ====================== 模型构建函数 ======================
def build_model(conv_filters, kernel_size, dropout_conv, dense_units, 
               dropout_dense, l2_reg, lr, vocab_size, embedding_dim, max_len,
               embedding_matrix=None):
    try:
        if embedding_matrix is not None:
            embedding_layer = Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                input_length=max_len,
                trainable=True  # 可以设置为False来冻结Embedding层
            )
        else:
            embedding_layer = Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_len,
                trainable=True
            )
        
        model = Sequential([
            embedding_layer,
            SpatialDropout1D(dropout_conv),
            Conv1D(conv_filters, kernel_size),
            BatchNormalization(),
            Activation('relu'),
            Dropout(dropout_conv),
            GlobalMaxPooling1D(),
            Dense(dense_units, activation='relu', kernel_regularizer=l2(l2_reg)),
            Dropout(dropout_dense),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"模型构建失败: {str(e)}")
        return None

# ====================== 数据预处理 ======================
def load_data(file_path):
    try:
        df = pd.read_excel(file_path)
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("数据文件必须包含text和label列")
        df = df.drop_duplicates(subset=['text']).dropna(subset=['text', 'label'])
        return df
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return None

def preprocess_data(df, test_size=0.2):
    if df is None:
        return None
    
    try:
        def cleantext(text):
            text = text.strip()
            text = re.sub(r'[^\u4e00-\u9fff\s！？，。]', ' ', text)
            return re.sub(r'\s+', ' ', text).strip()
        
        def random_deletion(words, p=0.1):
            return [w for w in words if random.random()>p] if len(words)>0 else words
        
        def random_swap(words, n_swaps=1):
            new = words.copy()
            for _ in range(n_swaps):
                if len(new)<2: break
                i,j = random.sample(range(len(new)), 2)
                new[i], new[j] = new[j], new[i]
            return new
        
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        df['cleantext'] = df['text'].apply(cleantext)
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['cleantext'], df['label'], 
            test_size=test_size, 
            stratify=df['label'],
            random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=0.5, 
            stratify=y_temp,
            random_state=42
        )
        
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(X_train, y_train):
            words = jieba.lcut(text)
            augmented_texts.append(" ".join(words))
            augmented_labels.append(label)
            
            deleted_words = random_deletion(words)
            if len(deleted_words) > 0:
                augmented_texts.append(" ".join(deleted_words))
                augmented_labels.append(label)
            
            swapped_words = random_swap(words)
            augmented_texts.append(" ".join(swapped_words))
            augmented_labels.append(label)
        
        X_train = pd.Series(augmented_texts)
        y_train = pd.Series(augmented_labels)
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'le': le
        }
    except Exception as e:
        print(f"数据预处理失败: {str(e)}")
        return None

# ====================== 实时更新图表的回调 ======================
class PlotCallBack(Callback):
    plot_data_ready = pyqtSignal(list, list, list, list, list, list)
    
    def __init__(self, result_window, log_window):
        super().__init__()
        self.result_window = result_window
        self.log_window = log_window
        self.val_losses = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []
        self.val_auc = []
    
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        
        self.val_losses.append(logs.get('val_loss', 0.0))
        self.val_accuracies.append(logs.get('val_accuracy', 0.0))
        
        # 将日志信息发送到日志窗口
        log_message = f"Epoch {epoch + 1}/{self.params['epochs']}\n"
        for key, value in logs.items():
            log_message += f"  {key}: {value:.4f}\n"
        
        self.log_window.append_log(log_message)
        
        # 计算验证集上的指标
        if hasattr(self, 'validation_data') and self.validation_data is not None:
            y_pred_prob = self.model.predict(self.validation_data[0])
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = self.validation_data[1]
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            roc_auc = roc_auc_score(label_binarize(y_true, classes=[0, 1, 2]), y_pred_prob, average='weighted')
            
            self.val_precisions.append(precision)
            self.val_recalls.append(recall)
            self.val_f1s.append(f1)
            self.val_auc.append(roc_auc)
            
            self.result_window.update_training_plot(
                self.val_losses, self.val_accuracies,
                self.val_precisions, self.val_recalls,
                self.val_f1s, self.val_auc
            )

# ====================== 重定向标准输出类 ======================
class StreamRedirector:
    def __init__(self, log_window):
        self.log_window = log_window

    def write(self, text):
        if text.strip():  # 只有在有非空文本时才添加日志
            self.log_window.append_log(text)
        sys.__stdout__.write(text)  # 同时将文本输出到控制台

    def flush(self):
        sys.__stdout__.flush()

# ====================== 训练线程 ======================
class TrainingThread(QThread):
    update_log = pyqtSignal(str)
    update_metrics = pyqtSignal(dict)
    training_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, config, data, result_window, log_window):
        super().__init__()
        self.config = config
        self.data = data
        self.tokenizer = Tokenizer()
        self.result_window = result_window
        self.log_window = log_window
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            self.update_log.emit("开始数据预处理...")
            
            if self.data is None:
                self.update_log.emit("数据预处理失败: 数据为空")
                return
            
            # 文本分词
            X_train_seg = self.data['X_train'].apply(lambda x: ' '.join(jieba.lcut(str(x))))
            X_val_seg = self.data['X_val'].apply(lambda x: ' '.join(jieba.lcut(str(x))))
            X_test_seg = self.data['X_test'].apply(lambda x: ' '.join(jieba.lcut(str(x))))
            
            # 构建词汇表
            self.tokenizer.fit_on_texts(X_train_seg)
            vocab_size = len(self.tokenizer.word_index) + 1
            
            # 序列填充
            max_len = 30
            X_train_seq = self.tokenizer.texts_to_sequences(X_train_seg)
            X_val_seq = self.tokenizer.texts_to_sequences(X_val_seg)
            X_test_seq = self.tokenizer.texts_to_sequences(X_test_seg)
            
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
            X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post')
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
            
            self.update_log.emit(f"词汇表大小: {vocab_size}")
            self.update_log.emit(f"训练集大小: {len(X_train_pad)}")
            self.update_log.emit(f"验证集大小: {len(X_val_pad)}")
            self.update_log.emit(f"测试集大小: {len(X_test_pad)}")
            
            # 加载预训练词向量
            embedding_matrix = None
            use_pretrained = self.config.get('use_pretrained_embeddings', False)
            if use_pretrained and 'embedding_file' in self.config:
                embedding_file = self.config['embedding_file']
                if embedding_file and os.path.exists(embedding_file):
                    self.update_log.emit(f"加载预训练词向量: {embedding_file}")
                    embedding_dim = self.config.get('embedding_dim', 300)
                    embedding_matrix = load_pretrained_embeddings(
                        embedding_file, self.tokenizer, embedding_dim
                    )
                    if embedding_matrix is not None:
                        self.update_log.emit(f"成功加载词向量，覆盖 {np.sum(np.sum(embedding_matrix, axis=1) != 0)} 个词")
                    else:
                        self.update_log.emit("预训练词向量加载失败，使用随机初始化")
                else:
                    self.update_log.emit("预训练词向量文件不存在，使用随机初始化")
            
            # 构建模型
            self.update_log.emit("构建模型中...")
            model = build_model(
                vocab_size=vocab_size,
                embedding_dim=self.config['embedding_dim'],
                max_len=30,
                conv_filters=self.config['conv_filters'],
                kernel_size=self.config['kernel_size'],
                dropout_conv=self.config['dropout_conv'],
                dense_units=self.config['dense_units'],
                dropout_dense=self.config['dropout_dense'],
                l2_reg=self.config['l2_reg'],
                lr=self.config['lr'],
                embedding_matrix=embedding_matrix
            )
            
            if model is None:
                self.update_log.emit("模型构建失败")
                return
            
            # 实时更新图表的回调
            plot_callback = PlotCallBack(self.result_window, self.log_window)
            
            # 训练模型
            self.update_log.emit("开始模型训练...")
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                min_lr=1e-6,
                patience=3,
                restore_best_weights=True
            )
            
            # 重定向标准输出和标准错误
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = StreamRedirector(self.log_window)
            sys.stderr = StreamRedirector(self.log_window)
            
            try:
                history = model.fit(
                    X_train_pad, self.data['y_train'],
                    validation_data=(X_val_pad, self.data['y_val']),
                    epochs=20,
                    batch_size=32,
                    callbacks=[early_stopping, reduce_lr, plot_callback],
                    verbose=1
                )
            finally:
                # 恢复标准输出和标准错误
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            
            # 发送训练过程数据
            self.update_log.emit("训练完成，正在评估模型...")
            
            # 评估模型
            y_pred_prob = model.predict(X_test_pad, verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            
            # 计算ROC曲线
            y_test_bin = label_binarize(self.data['y_test'], classes=[0, 1, 2])
            fpr, tpr, roc_auc = {}, {}, {}
            
            for i in range(3):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 计算平均AUC
            avg_auc = np.mean([roc_auc.get(i, 0.0) for i in range(3)])
            
            # 计算分类报告
            report = classification_report(
                self.data['y_test'], 
                y_pred, 
                target_names=self.data['le'].classes_,
                digits=4
            )
            self.update_log.emit("分类报告:\n" + report)
            
            # 发送最终指标
            self.update_metrics.emit({
                'accuracy': accuracy_score(self.data['y_test'], y_pred),
                'precision': precision_recall_fscore_support(self.data['y_test'], y_pred)[0].mean() if len(precision_recall_fscore_support(self.data['y_test'], y_pred)[0]) > 0 else 0.0,
                'recall': precision_recall_fscore_support(self.data['y_test'], y_pred)[1].mean() if len(precision_recall_fscore_support(self.data['y_test'], y_pred)[1]) > 0 else 0.0,
                'f1': precision_recall_fscore_support(self.data['y_test'], y_pred)[2].mean() if len(precision_recall_fscore_support(self.data['y_test'], y_pred)[2]) > 0 else 0.0,
                'roc_auc': avg_auc
            })
            
            self.training_finished.emit()
            self.update_log.emit("训练完成!")
            
        except Exception as e:
            self.error_occurred.emit(f"训练错误: {str(e)}")

# ====================== 主窗口 ======================
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = MainUI()
        self.ui.setupUi(self)
        self.setWindowTitle("情感分类分析系统")
        
        # 初始化子窗口
        self.result_window = ResultWindow(self)
        self.log_window = LogWindow(self)
        self.config_window = ConfigWindow(self)
        
        # 设置窗口属性
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.setFixedSize(900, 500)
        self.setStyleSheet("""
            QWidget {
                font-family: "Microsoft YaHei";
                font-size: 12px;
            }
            QPushButton {
                min-width: 120px;
                max-width: 120px;
                padding: 8px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        # 连接按钮信号
        self.ui.pushButton.clicked.connect(self.show_config)
        self.ui.pushButton_2.clicked.connect(self.show_result)
        self.ui.pushButton_3.clicked.connect(self.show_log)

        # 将窗口居中
        self.center_on_screen()

    def center_on_screen(self):
        screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def show_config(self):
        self.hide()
        self.config_window.show()

    def show_result(self):
        self.hide()
        self.result_window.show()

    def show_log(self):
        self.hide()
        self.log_window.show()
        
    def show_error(self, message):
        QtWidgets.QMessageBox.critical(
            self, "错误", 
            f"发生错误：\n{message}",
            QtWidgets.QMessageBox.Ok
        )

# ====================== 配置窗口 ======================
class ConfigWindow(QtWidgets.QWidget):
    config_selected = QtCore.pyqtSignal(dict)
    data_loaded = QtCore.pyqtSignal(dict)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.ui = ConfigUI()
        self.ui.setupUi(self)
        
        # 固定窗口尺寸
        self.setFixedSize(900, 500)

        # 初始化控件
        self.training_thread = None
        self.data = None
        self.status_label = QtWidgets.QLabel("就绪", self)
        self.dataset_path = self.ui.lineEdit  # 使用UI文件中定义的文本框
        self.embedding_path = self.ui.lineEdit_2  # 使用UI文件中定义的文本框
        
        # 连接信号
        self.ui.pushButton_4.clicked.connect(self.return_to_main)
        self.ui.pushButton.clicked.connect(self.select_dataset)
        self.ui.pushButton_2.clicked.connect(self.start_train)
        self.ui.pushButton_3.clicked.connect(self.stop_train)
        self.ui.pushButton_5.clicked.connect(self.select_embedding)

        # 将窗口居中
        self.center_on_screen()

    def center_on_screen(self):
        screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # 设置样式
        self.setStyleSheet("""
            QLineEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                padding: 6px;
            }
            QComboBox {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                padding: 6px;
            }
        """)

    def select_dataset(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择数据集", "", "Excel文件 (*.xlsx);;所有文件 (*)"
        )
        if file_path:
            self.dataset_path.setText(file_path)  # 设置文本框内容
            try:
                self.status_label.setText("加载数据中...")
                df = load_data(file_path)
                if df is None:
                    self.status_label.setText("数据加载失败")
                    return
                
                self.data = preprocess_data(df)
                if self.data is None:
                    self.status_label.setText("数据预处理失败")
                    return
                
                self.data_loaded.emit(self.data)
                self.status_label.setText(f"数据加载成功 ({len(df)} 条记录)")
                QtWidgets.QMessageBox.information(self, "成功", "数据加载成功!")
            except Exception as e:
                self.status_label.setText("数据加载失败")
                QtWidgets.QMessageBox.critical(self, "错误", f"数据加载失败: {str(e)}")

    def select_embedding(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择预训练词向量", "", "文本文件 (*.txt *.vec)"
        )
        if file_path:
            self.embedding_path.setText(file_path)  # 设置文本框内容

    def start_train(self):
        if not hasattr(self, 'data') or self.data is None:
            QtWidgets.QMessageBox.warning(
                self, "警告", 
                "请先选择有效的数据集文件！",
                QtWidgets.QMessageBox.Ok
            )
            return
            
        config_map = {
            'A': dict(
                conv_filters=64, kernel_size=3, dropout_conv=0.2,
                dense_units=64, dropout_dense=0.3, l2_reg=0.007, lr=3e-4,
                embedding_dim=300, use_pretrained_embeddings=False
            ),
            'B': dict(
                conv_filters=32, kernel_size=5, dropout_conv=0.15,
                dense_units=128, dropout_dense=0.25, l2_reg=0.002, lr=1e-3,
                embedding_dim=300, use_pretrained_embeddings=False
            ),
            'C': dict(
                conv_filters=128, kernel_size=3, dropout_conv=0.25,
                dense_units=32, dropout_dense=0.20, l2_reg=0.0005, lr=1e-4,
                embedding_dim=300, use_pretrained_embeddings=False
            )
        }
        
        config_key = self.ui.comboBox.currentText()[0]
        config = config_map.get(config_key, config_map['A'])
        
        # 检查是否使用预训练词向量
        embedding_file = self.embedding_path.text()
        if embedding_file and os.path.exists(embedding_file):
            config['use_pretrained_embeddings'] = True
            config['embedding_file'] = embedding_file
        
        self.training_thread = TrainingThread(config, self.data, self.main_window.result_window, self.main_window.log_window)
        if self.training_thread:
            self.training_thread.update_log.connect(
                self.main_window.log_window.append_log)
            self.training_thread.update_metrics.connect(
                self.main_window.result_window.update_final_metrics)
            self.training_thread.error_occurred.connect(
                self.main_window.show_error)
            self.training_thread.finished.connect(
                lambda: QtWidgets.QMessageBox.information(
                    self, "完成", 
                    "模型训练已完成！",
                    QtWidgets.QMessageBox.Ok
                )
            )
            self.training_thread.start()
        
    def stop_train(self):
        if self.training_thread:
            self.training_thread.stop()

    def return_to_main(self):
        self.hide()
        self.main_window.show()

# ====================== 结果窗口 ======================
class ResultWindow(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.ui = ResultUI()
        self.ui.setupUi(self)
        
        # 固定窗口尺寸
        self.setFixedSize(900, 500)
        
        # 连接返回按钮
        self.ui.pushButton.clicked.connect(self.return_to_main)
        
        # 将窗口居中
        self.center_on_screen()

    def center_on_screen(self):
        screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

        # 设置样式
        self.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        
        # 创建图表区域
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout = QtWidgets.QVBoxLayout(self.ui.frame_3)
        layout.addWidget(self.canvas)
        
        # 初始化图表
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("模型训练与评估结果")
        self.ax.set_xlabel("训练轮次")
        self.ax.set_ylabel("指标值")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # 训练指标曲线
        self.val_loss_line, = self.ax.plot([], [], 'r-', linewidth=0.5, label='验证损失')
        self.val_acc_line, = self.ax.plot([], [], 'b-', linewidth=0.5, label='验证准确率')
        self.val_prec_line, = self.ax.plot([], [], 'g-', linewidth=0.5, label='验证精确率')
        self.val_rec_line, = self.ax.plot([], [], 'c-', linewidth=0.5, label='验证召回率')
        self.val_f1_line, = self.ax.plot([], [], 'm-', linewidth=0.5, label='验证F1分数')
        self.val_auc_line, = self.ax.plot([], [], 'y-', linewidth=0.5, label='验证AUC')
        
        # 图例
        self.ax.legend(loc='lower right')

    def update_training_plot(self, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s, val_auc):
        """更新训练图表"""
        if not val_losses:
            return  # 如果数据为空，直接返回
            
        epochs = np.arange(len(val_losses))
        
        self.val_loss_line.set_xdata(epochs)
        self.val_loss_line.set_ydata(val_losses)
        
        self.val_acc_line.set_xdata(epochs)
        self.val_acc_line.set_ydata(val_accuracies)
        
        self.val_prec_line.set_xdata(epochs)
        self.val_prec_line.set_ydata(val_precisions)
        
        self.val_rec_line.set_xdata(epochs)
        self.val_rec_line.set_ydata(val_recalls)
        
        self.val_f1_line.set_xdata(epochs)
        self.val_f1_line.set_ydata(val_f1s)
        
        self.val_auc_line.set_xdata(epochs)
        self.val_auc_line.set_ydata(val_auc)
        
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def update_final_metrics(self, metrics):
        """更新最终评估指标"""
        if metrics is None:
            return
            
        self.ax.clear()
        self.ax.set_title("模型最终评估结果")
        self.ax.set_xlabel("指标")
        self.ax.set_ylabel("值")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        metrics_names = ['准确率', '精确率', '召回率', 'F1', 'AUC']
        metrics_values = [
            metrics.get('accuracy', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1', 0.0),
            metrics.get('roc_auc', 0.0)
        ]
        
        self.ax.bar(metrics_names, metrics_values, color=['b', 'g', 'c', 'm', 'y'])
        for i, v in enumerate(metrics_values):
            self.ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        self.canvas.draw_idle()

    def update_roc(self, roc_data):
        """更新ROC曲线"""
        # 清除现有ROC曲线
        self.ax.clear()
        self.ax.set_title("模型训练与评估结果")
        self.ax.set_xlabel("假阳性率")
        self.ax.set_ylabel("真阳性率")
        self.ax.grid(True, linestyle='--', alpha=0.7)

        # 绘制训练指标曲线
        if hasattr(self, 'val_loss_line'):
            self.ax.plot(self.val_loss_line.get_xdata(), self.val_loss_line.get_ydata(), 'r-', linewidth=0.5, label='验证损失')
            self.ax.plot(self.val_acc_line.get_xdata(), self.val_acc_line.get_ydata(), 'b-', linewidth=0.5, label='验证准确率')
            self.ax.plot(self.val_prec_line.get_xdata(), self.val_prec_line.get_ydata(), 'g-', linewidth=0.5, label='验证精确率')
            self.ax.plot(self.val_rec_line.get_xdata(), self.val_rec_line.get_ydata(), 'c-', linewidth=0.5, label='验证召回率')
            self.ax.plot(self.val_f1_line.get_xdata(), self.val_f1_line.get_ydata(), 'm-', linewidth=0.5, label='验证F1分数')
            self.ax.plot(self.val_auc_line.get_xdata(), self.val_auc_line.get_ydata(), 'y-', linewidth=0.5, label='验证AUC')
        
        if roc_data and 'classes' in roc_data:
            for i in range(len(roc_data['classes'])):
                class_name = roc_data['classes'][i]
                fpr = roc_data.get('fpr', [None])[i]
                tpr = roc_data.get('tpr', [None])[i]
                roc_auc = roc_data.get('roc_auc', [0.0])[i]
                
                if fpr is not None and tpr is not None:
                    self.ax.plot(fpr, tpr, lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.4f})')
        
        # 更新图例
        self.ax.legend(loc='lower right')
        
        # 重绘图表
        self.canvas.draw_idle()

    def return_to_main(self):
        self.hide()
        self.main_window.show()

# ====================== 日志窗口 ======================
class LogWindow(QtWidgets.QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.ui = LogUI()
        self.ui.setupUi(self)
        
        # 固定窗口尺寸
        self.setFixedSize(900, 500)
        
        # 初始化日志文本框
        self.text_edit = QtWidgets.QTextEdit(self.ui.frame_2)
        self.text_edit.setGeometry(10, 10, 780, 580)
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #F5F5F5;
                border: 1px solid #CCCCCC;
                padding: 8px;
                font-family: Consolas;
            }
        """)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)  # 始终显示垂直滚动条
        
        # 连接返回按钮
        self.ui.pushButton.clicked.connect(self.return_to_main)

        # 将窗口居中
        self.center_on_screen()

    def center_on_screen(self):
        screen_geometry = QtWidgets.QDesktopWidget().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)

    def append_log(self, text):
        if text is not None:
            self.text_edit.append(text)
            self.text_edit.verticalScrollBar().setValue(
                self.text_edit.verticalScrollBar().maximum()
            )

    def return_to_main(self):
        self.hide()
        self.main_window.show()

if __name__ == "__main__":
    # 启用高清屏支持
    QtWidgets.QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QtWidgets.QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())