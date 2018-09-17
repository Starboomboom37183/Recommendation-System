#-*-coding:utf-8-*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import  win32gui
import  win32api
import  win32con
import  SendKeys
import time
from ctypes import *
from pymouse import PyMouse
from pykeyboard import PyKeyboard

m = PyMouse()
k = PyKeyboard()
#多次登录，传入账号密码
def qqLoad(qq,pwd):
    #使用系统模块os，打开qq
    #必须是单引号+双引号才能运行
    cwd = u"C:\\Program Files (x86)\\黑桃棋牌\\启动游戏.exe"
    ##cwd =  u"E:\\Program Files (x86)\\Tencent\\QQ\\Bin\\QQScLauncher.exe"
    ##cwd = "notepad.exe"
    os.startfile(cwd)
    #留给qq界面点响应时间
    time.sleep(3)
    # 获取窗口的句柄，参数1：类名，参数2：标题
    handle = win32gui.FindWindow(None,u"黑桃棋牌")
    print handle
    #返回指定窗口的显示状态以及被恢复的、最大化的和最小化的窗口位置
    logId = win32gui.GetWindowPlacement(handle)
    print logId
    #设置鼠标位置,横坐标等于左上角数加输入框离左边界的差值，纵坐标等于左上角数加输入框离上边界的差值
    #差值可用截图工具，测量像素差值
    windll.user32.SetCursorPos(465,467)
    #模拟鼠标点击操作,左键先按下，再松开
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)
    #！！注意，必须要延时，才能正确输入，否则输入内容错误
    time.sleep(0.2)
    #安装SendKeys库，可自动输入内容
    k.press_key(k.shift_l_key)
    k.release_key(k.shift_l_key)
    SendKeys.SendKeys(qq)
    time.sleep(0.3)
    #按下tab键，切换到输入密码
    #模拟键盘操作，查看键盘对应asc码，tab键对应asc码是9
    #先按下，再松开
    win32api.keybd_event(9,0,0,0)
    win32api.keybd_event(9,0,win32con.KEYEVENTF_KEYUP,0)
    time.sleep(0.3)
    #输入密码，点击回车键登录
    ##print pwd
    ##k.release_key()
    ##k.press_keys(pwd)
    SendKeys.SendKeys(pwd)
    time.sleep(0.3)
    win32api.keybd_event(13,0,0,0)
    win32api.keybd_event(13,0,win32con.KEYEVENTF_KEYUP,0)
    return handle
def click_into_baijiale():
    ##m.click(116,279,2,1)
    time.sleep(1)
    ##m.move(116,279)
    windll.user32.SetCursorPos(116, 279)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    ##
    time.sleep(0.5)
    ##windll.user32.SetCursorPos(160, 342)
    m.move(160, 342)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

##
def printScreen():
    return

def xiazhu(s,hwnd):
    time.sleep(0.5)
    ##win32gui.SetForegroundWindow(hwnd)
    windll.user32.SetCursorPos(673, 350)
    hdc = win32gui.GetDC(hwnd)
    color1=color2 = win32gui.GetPixel(hdc,673,350)
    win32gui.ReleaseDC(hwnd,hdc)
    while color1==color2 and color1==2787168:
        time.sleep(1)
        hdc = win32gui.GetDC(hwnd)
        color2 = win32gui.GetPixel(hdc, 673, 350)
        win32gui.ReleaseDC(hwnd, hdc)






def xiazhu1(s,hwnd):
    time.sleep(0.5)

    ##win32gui.SetForegroundWindow(hwnd)


    windll.user32.SetCursorPos(671, 336)
    hdc = win32gui.GetDC(hwnd)
    color = win32gui.GetPixel(hdc,671,336)
    win32gui.ReleaseDC(hwnd,hdc)


    ##x,y = 673,328
    ##windll.user32.SetCursorPos(x, y)


    ##windll.user32.SetCursorPos(s[0],s[1])
    ##win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    ##win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)







if __name__ == '__main__':
    #在文件中读取帐号密码信息

    #循环打开每一行，使用split分成列表

    qq_no = 'yongwang260'
    qq_pwd = '19930418'
    hwnd = qqLoad(qq_no,qq_pwd)
    click_into_baijiale()
    s1 = (582,579)
    s2 = (655,584)
    s3 = (723,585)
    xiazhu1(s1,hwnd)

