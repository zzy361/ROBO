FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    $PIP_INSTALL \
        PyMySQL \
        scrapy \
        moment \
        beautifulsoup4 \
        sklearn \
        sklearn2 \
        pandas_datareader \
        xlrd \
        openpyxl \
        flask \
        xlrd \
        xlsxwriter \
        xlwt \
        xlutils \
        gym \
        pymssql \
        scipy \
        scikit-learn \
        seaborn \
        xgboost \
        lightgbm \
        eli5 \
        oss2 \
        traceback2 \
        python-dateutil \
        sqlalchemy \
        Pyomo==5.6.1 \
        keras==2.1.2 \
        mysql-connector \
	pandas==0.24.2 \
        matplotlib \
        talib-binary && \
    rm -f /etc/localtime && \
    ln -sf /usr//local/lib/python3.6/dist-packages/pytz/zoneinfo/Asia/Shanghai /etc/localtime
