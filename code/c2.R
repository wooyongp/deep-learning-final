library(tidyverse)
library(data.table)
library(haven)

# h__0141(location), h__0601(divorce), h__0150(num member), p_5502(marstat change) p_5504(marstat changeD)
p_var <- c('1642', '5501', '5502', '5504')
h_var <- c('0141', '0601', '0150')

for (i in 3:25) {
if (i < 10) {
  assign(paste0('d0', i), as.data.table(read_dta(paste0('data/klips/klips0', i, 'p.dta'))[,paste0("p0", i, p_var)])) #%>% 
        #    left_join(as.data.table(read_dta(paste0('data/klips/klips0', i, 'h.dta'))), by=paste0('hhid0', i)))
} else {
  assign(paste0('d', i), as.data.table(read_dta(paste0('data/klips/klips', i, 'p.dta'))[,paste0("p", i, p_var)])) #%>% 
        #    left_join(fread(paste0('data/klips/klips', i, 'h.dta')), by=paste0('hhid', i)))
    }
}


data <- read_dta(paste0('data/klips/klips0', 1, 'p.dta')) %>% as.data.table()
data %>% view()


d22 <- read_dta('data/klips/klips22p.dta') %>% 
  left_join(read_dta('data/klips/klips22h.dta'), by='hhid22')

d23 <- read_dta('data/klips/klips23p.dta') %>% 
  left_join(read_dta('data/klips/klips23h.dta'), by='hhid23')

d24 <- read_dta('data/klips/klips24p.dta') %>% 
  left_join(read_dta('data/klips/klips24h.dta'), by='hhid24')

d25 <- read_dta('data/klips/klips25p.dta') %>% 
  left_join(read_dta('data/klips/klips25h.dta'), by='hhid25')

# h__0141(location), h__0601(divorce), h__0150(num member), p_5502(marstat change) p_5504(marstat changeD)
