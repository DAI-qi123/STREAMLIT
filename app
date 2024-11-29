import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体，确保表格和图形中的中文都能正确显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei，支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 常见的回归模型
models = {
    "线性回归": LinearRegression,
    "决策树": DecisionTreeRegressor,
    "随机森林": RandomForestRegressor,
    "梯度提升": GradientBoostingRegressor,
    "支持向量回归": SVR,
    "K 最近邻": KNeighborsRegressor,
    "神经网络 (MLP)": MLPRegressor,
    "高斯过程": GaussianProcessRegressor,
    "岭回归": Ridge,
    "Lasso 回归": Lasso,
}

st.sidebar.title("选项")

# 主选择框
option = st.sidebar.selectbox("请选择操作", ["训练模型", "预测结果", "数据分析"])

if option == "训练模型":
    model_filename = ""
    st.title("训练机器学习模型")
    uploaded_file = st.file_uploader("上传数据集 (Excel 格式)", type=["xlsx", "xls"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("数据集预览：", df.head())

        features = st.multiselect("选择特征列", df.columns)
        label = st.selectbox("选择标签列", df.columns)

        if features and label:
            X = df[features]
            y = pd.to_numeric(df[label], errors='coerce')  # 转换目标列为数值类型

            # 处理目标变量中的 NaN 值
            if y.isna().any():
                st.warning("目标列包含无效值或缺失值，已删除对应行。")
                X = X[~y.isna()]
                y = y.dropna()

            normalize = st.radio("是否进行归一化处理", ["不归一化", "标准化 (StandardScaler)", "归一化 (MinMaxScaler)"])
            if normalize == "标准化 (StandardScaler)":
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            elif normalize == "归一化 (MinMaxScaler)":
                scaler = MinMaxScaler()
                X = scaler.fit_transform(X)

            model_name = st.selectbox("选择回归算法", list(models.keys()))
            model_class = models[model_name]
            st.subheader(f"{model_name} 超参数设置")
            params = {}
            if model_name == "线性回归":
                st.write("此算法没有可调节的超参数。")
            elif model_name == "决策树":
                params["max_depth"] = st.number_input("最大深度 (max_depth)", 1, 100, 10)
                params["min_samples_split"] = st.number_input("最小样本分割数 (min_samples_split)", 2, 100, 2)
                params["min_samples_leaf"] = st.number_input("最小叶子节点样本数 (min_samples_leaf)", 1, 100, 1)
            elif model_name == "随机森林":
                params["n_estimators"] = st.number_input("树的数量 (n_estimators)", 10, 500, 100)
                params["max_depth"] = st.number_input("最大深度 (max_depth)", 1, 100, 10)
                params["min_samples_split"] = st.number_input("最小样本分割数 (min_samples_split)", 2, 100, 2)
                params["min_samples_leaf"] = st.number_input("最小叶子节点样本数 (min_samples_leaf)", 1, 100, 1)
            elif model_name == "梯度提升":
                params["n_estimators"] = st.number_input("树的数量 (n_estimators)", 10, 500, 100)
                params["learning_rate"] = st.slider("学习率 (learning_rate)", 0.01, 1.0, 0.1)
                params["max_depth"] = st.number_input("最大深度 (max_depth)", 1, 100, 10)
                params["min_samples_split"] = st.number_input("最小样本分割数 (min_samples_split)", 2, 100, 2)
                params["min_samples_leaf"] = st.number_input("最小叶子节点样本数 (min_samples_leaf)", 1, 100, 1)

            if st.button("训练模型"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = model_class(**params)
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)

                # 显示 R² 分数
                st.write(f"训练集 R²: {r2_score(y_train, train_pred):.4f}")
                st.write(f"测试集 R²: {r2_score(y_test, test_pred):.4f}")

                # 绘制预测散点图
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].scatter(y_train, train_pred, alpha=0.7)
                ax[0].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
                ax[0].set_title("训练集实际值 vs 预测值")
                ax[0].set_xlabel("实际值")
                ax[0].set_ylabel("预测值")

                ax[1].scatter(y_test, test_pred, alpha=0.7, color='orange')
                ax[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
                ax[1].set_title("测试集实际值 vs 预测值")
                ax[1].set_xlabel("实际值")
                ax[1].set_ylabel("预测值")

                st.pyplot(fig)

                # 获取桌面路径并保存模型
                desktop_path = str(Path.home() / "Desktop")
                pkl_filename = os.path.join(desktop_path, "trained_model.pkl")

                with open(pkl_filename, 'wb') as f:
                    pickle.dump({"model": model}, f)
                st.success(f"模型已保存到桌面: {pkl_filename}")

elif option == "预测结果":
    st.title("预测结果")
    uploaded_model_file = st.file_uploader("上传训练好的模型文件 (PKL 格式)", type=["pkl"])
    if uploaded_model_file:
        data = pickle.load(uploaded_model_file)
        trained_model = data["model"]
        prediction_data_file = st.file_uploader("上传待预测数据 (Excel 格式)", type=["xlsx", "xls"])
        if prediction_data_file:
            pred_df = pd.read_excel(prediction_data_file)

            # 显示文件预览
            st.write("数据预览：", pred_df.head())

            # 让用户选择要用于预测的列
            selected_columns = st.multiselect("选择用于预测的特征列", pred_df.columns)

            # 如果没有选择特征列，则提示用户
            if not selected_columns:
                st.warning("请选择用于预测的特征列。")
            else:
                # 只使用用户选择的特征列
                pred_df_selected = pred_df[selected_columns]

                # 清洗数据：将非数值型的列排除
                pred_df_selected = pred_df_selected.apply(pd.to_numeric, errors='coerce')

                # 检查是否存在任何缺失值，并删除包含缺失值的行
                pred_df_selected = pred_df_selected.dropna()

                if not pred_df_selected.empty:
                    predictions = trained_model.predict(pred_df_selected)
                    st.write("预测结果：", predictions)
                else:
                    st.warning("预测数据中包含无法转换为数字的值或缺失值，无法进行预测。")

elif option == "数据分析":
    st.title("数据分析工具")
    uploaded_file = st.file_uploader("上传数据集 (Excel 格式)", type=["xlsx", "xls"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("数据集预览：", df.head())
        analysis_option = st.radio("选择分析功能", ["变量聚类", "相关性分析", "灰色关联分析", "随机森林变量重要性"])

        if analysis_option == "变量聚类":
            st.subheader("变量聚类")
            features = st.multiselect("选择用于聚类的变量", df.columns)
            cluster_method = st.selectbox("选择聚类方法", ["K-Means", "层次聚类"])  # 移除 DBSCAN
            n_clusters = st.number_input("聚类数 (仅适用于 K-Means 和层次聚类)", min_value=2, max_value=10, value=3)

            if features and st.button("开始聚类"):
                X = df[features].dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # PCA 降维到 2D
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                if cluster_method == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42)
                elif cluster_method == "层次聚类":
                    model = AgglomerativeClustering(n_clusters=n_clusters)

                labels = model.fit_predict(X_scaled)
                df["Cluster"] = labels
                st.write("聚类结果：", df)

                # 可视化聚类分布
                plt.figure(figsize=(8, 6))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50)
                plt.title("聚类分布图 (PCA 降维)", fontsize=14)
                plt.xlabel("PCA 1")
                plt.ylabel("PCA 2")
                st.pyplot(plt)

        elif analysis_option == "相关性分析":
            st.subheader("相关性分析")
            selected_cols = st.multiselect("选择用于相关性分析的变量", df.columns)
            correlation_type = st.radio("选择相关性分析方法", ["皮尔逊相关性", "斯皮尔曼相关性"])
            method = "pearson" if correlation_type == "皮尔逊相关性" else "spearman"

            if selected_cols:
                correlation_matrix = df[selected_cols].corr(method=method)
                # 格式化相关性矩阵
                correlation_matrix = correlation_matrix.applymap(lambda x: int(x) if x.is_integer() else round(x, 2))

                st.write(f"{correlation_type}矩阵：", correlation_matrix)

                # 显示完整数值的热力图
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(
                    correlation_matrix,
                    annot=True,
                    fmt=".2f",  # 显示两位小数
                    cmap="coolwarm",
                    ax=ax
                )
                ax.set_title(f"{correlation_type}相关性热力图", fontsize=14)
                st.pyplot(fig)

        elif analysis_option == "灰色关联分析":
            st.subheader("灰色关联分析")
            target_col = st.selectbox("选择目标变量列", df.columns)
            feature_cols = st.multiselect("选择特征变量列", [col for col in df.columns if col != target_col])
            if target_col and feature_cols:
                X = df[feature_cols]
                y = df[target_col]
                normalized_X = (X - X.min()) / (X.max() - X.min())
                normalized_y = (y - y.min()) / (y.max() - y.min())
                grey_relation = normalized_X.apply(lambda x: 1 - abs(x - normalized_y).sum() / len(normalized_y))
                grey_relation_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Grey Relation": grey_relation.values
                }).sort_values(by="Grey Relation", ascending=False)

                st.write("灰色关联度：", grey_relation_df)

                # 可视化灰色关联度条形图
                fig, ax = plt.subplots()
                sns.barplot(x="Grey Relation", y="Feature", data=grey_relation_df, ax=ax)
                ax.set_title("灰色关联分析结果")
                st.pyplot(fig)

        elif analysis_option == "随机森林变量重要性":
            st.subheader("随机森林变量重要性")
            target_col = st.selectbox("选择目标变量列", df.columns)
            feature_cols = st.multiselect("选择特征变量列", [col for col in df.columns if col != target_col])
            if target_col and feature_cols:
                X = df[feature_cols]
                y = df[target_col]
                model = RandomForestRegressor(random_state=42)
                model.fit(X, y)
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": feature_cols,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)

                st.write("变量重要性：", importance_df)
                fig, ax = plt.subplots()
                sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
                ax.set_title("随机森林变量重要性分析")
                st.pyplot(fig)
