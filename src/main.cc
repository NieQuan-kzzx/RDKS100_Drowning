#include <QApplication>
#include "mainwindow.h"
#include "gflags/gflags.h"

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    QApplication a(argc, argv);
    
    MainWindow w;
    w.show();
    
    return a.exec();
}