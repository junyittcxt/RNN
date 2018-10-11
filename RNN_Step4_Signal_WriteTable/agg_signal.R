library(RMySQL)

setwd("C:/Users/Workstation/Desktop/RNN/RNN_Step4_Signal_WriteTable")
file_name  <- dir(pattern = "^rnn_signal")[1]

temp_str <- strsplit(file_name, "[.]")[[1]][1]
instrument_name <- strsplit(temp_str, "[_]")[[1]][3]

instrument_name <- tolower(instrument_name)

# df2 <- read.csv("aggregate_signal.csv", stringsAsFactors = F)
df2 <- read.csv(file_name, stringsAsFactors = F)
# df2$Date <- as.Date(df2$Date)
# colnames(df2) <- c("Date", "actual_class", "signal_raw", "signal_class")

mydb = dbConnect(MySQL(), user='cxtanalytics', password='3.1415cxt', dbname='algotrading', host='192.168.1.110')
# dbListTables(mydb)
write_okay <- dbWriteTable(conn = mydb, value = df2, name = paste0("mlsignal_rnn_min_",instrument_name), row.names = F, overwrite = T)
print(write_okay)
disconnect_msg <- dbDisconnect(mydb)

print("OKAY!")
print("Closing in 3 seconds...")

file.remove(file_name)

ff  <- dir(pattern = "^z_latest")[1]
file.rename(from = ff, to = paste0("z_latest_rnn_", file_name, ".txt"))

Sys.sleep(3)



write_okay <- dbWriteTable(conn = mydb, value = df2, name = paste0("mlsignal_rnn_min_",instrument_name), row.names = F, overwrite = T)




rs = dbSendQuery(mydb, "select * from mlsignal_rnn_min_eurusd")
data = fetch(rs, n=-1)

