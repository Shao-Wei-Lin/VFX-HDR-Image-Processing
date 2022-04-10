### Python package setting

- rawpy == 0.14.0
- numpy == 1.18.2
- pyexr == 0.3.7

### Usage of our code

```bash
python hw1.py [-h] [-a, --alignment] [-d, --dodge_type]
```
其中參數代表意思如下：
+ [-h]：查看幫助。
+ [-a]：若設置，則讀完檔案後會先做alignment再進行assemble HDR。
+ [-d]：實現 automatic dodging-and-burning 的方式，共有兩種，在底下 tone-mapping 的分歧點2有詳細說明。設成1則為該段前者的實作方式，設成2則為後者的實作方式。預設值為1。

舉例而言，若採用 alignment 且採預設實作 automatic dodging-and-burning 的方式，執行方式如下：

```bash 
python hw1.py -a
```



