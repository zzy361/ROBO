#!/bin/sh

IMAGE_NAME="976564756732.dkr.ecr.ap-northeast-1.amazonaws.com/thiztech/robo-advisor"
IMAGE_TAG="base-20190401"

# login aws ecr server
export AWS_ACCESS_KEY_ID=AKIA6GX6LUD6JSND3U4S
export AWS_SECRET_ACCESS_KEY=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
export AWS_DEFAULT_REGION=ap-northeast-1

$(aws ecr get-login --no-include-email --region ap-northeast-1)

# remove local docker images
docker ps -aqf ancestor=${IMAGE_NAME}:${IMAGE_TAG} | xargs -n 1 docker rm -f
docker images -qa ${IMAGE_NAME}:${IMAGE_TAG} | xargs -n 1 docker rmi -f
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

# push image
docker push ${IMAGE_NAME}:${IMAGE_TAG}
