
from functools import wraps
import sys, os
import logging
import traceback2

def log_email():
    def decorator(func):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(filename=os.path.split(__file__)[1] + '.log', encoding='utf-8', mode='a'),logging.StreamHandler(sys.stdout)],
                            format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                            datefmt='%m%d_%H:%M:%S')

        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        from email.header import Header

        emailSender = 'tommy.wei@thiztech.com'
        senderPassword = 'WEIxiang19900906'

        emailReceiver = ['mars.ma@thizgroup.com', 'tommy.wei@thizgroup.com', 'andrew.wang@thizgroup.com','jason.han@thizgroup.com']
        senderUsername = emailSender
        senderServer = 'smtp.exmail.qq.com'
        @wraps(func)
        def call_func(*args,**kwargs):
            logging.info(u'============================开始运行===============================')

            try:
                result = func(*args,**kwargs)
            except Exception as e:
                sender = emailSender
                receivers = emailReceiver
                message = MIMEMultipart()
                message['From'] = Header(u"定时任务", 'utf-8')
                message['To'] = Header("", 'utf-8')
                subject = u"定时任务" + os.path.split(__file__)[1]
                message['Subject'] = Header(subject, 'utf-8')
                errorinfo = u'异常信息:\n' + str(traceback2.format_exc())
                message.attach(MIMEText(errorinfo, 'plain', 'utf-8'))
                for i in range(100):
                    try:
                        smtpObj = smtplib.SMTP_SSL(senderServer, 465)
                        smtpObj.login(senderUsername, senderPassword)
                        smtpObj.sendmail(sender, receivers, message.as_string())
                        logging.info(u"邮件发送成功")
                        break
                    except smtplib.SMTPException as e:
                        logging.error(e)
                logging.error(errorinfo)
                logging.info(u'============================运行错误===============================')
                exit(0)
            logging.info(u'============================正常结束===============================')

            return result
        return call_func
    return decorator
