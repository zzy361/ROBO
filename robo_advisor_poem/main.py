from log_email import log_email
from config import g
g.init()

@log_email()
def main():
    import daily_out

main()
