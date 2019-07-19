FROM 976564756732.dkr.ecr.ap-northeast-1.amazonaws.com/thiztech/robo-advisor:base-20190401
COPY *.py /app/
COPY ai /app/ai/
COPY market_view /app/market_view/
COPY multifactor /app/multifactor/
CMD ["python", "/app/main.py"]
