#!/usr/bin/env bash

echo $1 $2 $3
#export PORT=8082
#read -p '-p'

#exit
function help(){
    echo "生成身份证样本"
    echo "命令格式："
    echo "  img_gen.sh --port|-p [默认8080] --worker|-w [默认3]"
    echo "  server.sh stop"
    exit
}

if [ "$1" = "stop" ]; then
    echo "停止 Tesseract OCR Web 服务"
    ps aux|grep ocr_web_server|grep -v grep|awk '{print $2}'|xargs kill -9
    exit
fi


if [ -z "$*" ]; then
    help
    exit
fi

#if [ "$1" = "debug" ]; then
#    echo "OCR Web 服务调试模式"
#    gunicorn --workers=1 --name=ocr_web_server --bind=0.0.0.0:8080 --timeout=300 server:app
#    exit
#fi

if [ ! "$1" = "start" ]; then
    help
    exit
fi

echo "启动 OCR Web 服务器..."
Date=$(date +%Y%m%d%H%M)
ARGS=`getopt -o p: --long port: -n 'help.bash' -- "$@"`
eval set -- "${ARGS}"
echo  '$2:' $2
echo 'args:' ${ARGS}
#exit
while true ;
do
        case "$2" in
                -p|--port)
                    echo "自定义端口号：$2"
                    if ! [ $PORT ]; then PORT=$2; fi #如果已经在环境变量中定义了，则不覆盖，环境变量优先级最大！！！这个是为了兼容容器方式启动，因为容器方式只能通过环境变量传入这些参数
                    shift 2
                    ;;

                --) shift ; break ;;
                *) help; exit 1 ;;
        esac
done

if [ $? != 0 ]; then
    help
    exit 1
fi

if [ $? != 0 ]; then
    help
    exit 1
fi


echo "服务器Single模式启动..."
# 参考：https://medium.com/building-the-system/gunicorn-3-means-of-concurrency-efbb547674b7
# worker=3是根据GPU的显存数调整出来的，ration=0.2，大概一个进程占满为2.5G,4x2.5=10G显存
_CMD="gunicorn --name=tesseract_ocr_web_server --workers=3 --bind 0.0.0.0:$PORT \
    --worker-class=gevent --worker-connections=10 --timeout=300 \
    server.server:app"
echo "启动服务："
echo "$_CMD"
eval $_CMD
exit 0

