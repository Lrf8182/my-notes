[利用 SSH 密钥链接 GitHub – SHUSCT Wiki](https://shusct.github.io/wiki/blog/connect-to-github-with-ssh-keys/)

**把在github上创建的repo clone到服务器上（git clone)	**

clone 因为 repo 的 http url 被墙了所以用 ssh 连接，那你就要一个能连接到你的账户的 ssh 密钥才行

ssh 就是链接 github 的一种方式
有一个公钥有一个私钥，两个文件
你想和 github 服务器建立链接，你就要本地和github有匹配的一对密钥


#### 如果不慎删除内容并退出

![b8c1b45df92c0660d96de135fe8f2e88.png](:/30f1814676c144aa8321d0e912adf282)  
R之后 ， WQ，再进去还是没有恢复  
**答：**  
先 ls -a  
删除 .lrf.pub.swp

#### vim 无法复制文件

**答：**  
你要复制文件lrf.pub  
就输入命令  
cat lrf.pub  
文件里的所有东西就都输出到终端了