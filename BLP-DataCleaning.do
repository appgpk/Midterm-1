clear
cd "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1"

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////1.BRESHAPING THE DATA////////////////////////////
////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////1.1 Share ////////////////////////////////

import delimited "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/midterm_simulated_market_data_s.csv",clear
rename md share1
rename th share2
rename sb share3
reshape long share, i(marketid) j(firmid)
gen id = firmid
save "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/s.dta", replace

///////////////////////////////1.2 Price & Caracteristic ////////////
import delimited "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/midterm_simulated_market_data_x.csv", clear 
rename price_md price1 
rename price_th price2
rename price_sb price3
rename caffeine_score_md caffeine_score1
rename caffeine_score_th caffeine_score2
rename caffeine_score_sb caffeine_score3
reshape long price caffeine_score, i(marketid) j(firmid)
gen id = firmid
save "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/x.dta", replace

///////////////////////////////1.3 Demand instrument ///////////////////////////

import delimited "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/midterm_simulated_market_data_zd.csv",clear
rename zd1_md zd11
rename zd1_th zd12
rename zd1_sb zd13
rename zd2_md zd21
rename zd2_th zd22
rename zd2_sb zd23
rename zd3_md zd31
rename zd3_th zd32
rename zd3_sb zd33
rename zd4_md zd41
rename zd4_th zd42
rename zd4_sb zd43
rename zd5_md zd51
rename zd5_th zd52
rename zd5_sb zd53
rename zd6_md zd61
rename zd6_th zd62
rename zd6_sb zd63
rename zd7_md zd71
rename zd7_th zd72
rename zd7_sb zd73
reshape long zd1 zd2 zd3 zd4 zd5 zd6 zd7, i(marketid) j(firmid)
gen id = firmid
save "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/zd.dta", replace

///////////////////////////////1.4 Supply instrument ///////////////////////////

import delimited "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/midterm_simulated_market_data_zs.csv",clear
rename zs1_md zs11
rename zs1_th zs12
rename zs1_sb zs13
rename zs2_md zs21
rename zs2_th zs22
rename zs2_sb zs23
rename zs3_md zs31
rename zs3_th zs32
rename zs3_sb zs33
rename zs4_md zs41
rename zs4_th zs42
rename zs4_sb zs43
rename zs5_md zs51
rename zs5_th zs52
rename zs5_sb zs53
rename zs6_md zs61
rename zs6_th zs62
rename zs6_sb zs63
rename zs7_md zs71
rename zs7_th zs72
rename zs7_sb zs73
reshape long zs1 zs2 zs3 zs4 zs5 zs6 zs7, i(marketid) j(firmid)
gen id = firmid
save "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/zs.dta", replace


///////////////////////////////1.5 Cost attributes /////////////////////////////

import delimited "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/midterm_simulated_market_data_w.csv",clear
egen caffeine_scoreAM = rmin(caffeine_score_md caffeine_score_th)

gen caffeine_scoreAM2 = (caffeine_score_md+caffeine_score_th)*0.5
rename caffeine_score_md costattributes1
rename caffeine_score_th costattributes2
rename caffeine_score_sb costattributes3

reshape long costattributes, i(marketid) j(firmid)
replace caffeine_scoreAM = costattributes if firmid == 3
replace caffeine_scoreAM2 = costattributes if firmid == 3
gen id = firmid
save "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/w.dta", replace

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////// 2.MERGING THE DATA /////////////////////////////
////////////////////////////////////////////////////////////////////////////////


///////////////////////////////2.1 Combined data ///////////////////////////////

use "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/s.dta",clear
merge m:m  marketid firmid id using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/x.dta"
drop _merge
merge m:m  marketid firmid id using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/zd.dta"
drop _merge
merge m:m  marketid firmid id using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/zs.dta"
drop _merge
merge m:m  marketid firmid id using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/w.dta"
drop _merge
export delimited using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/MIDTERM1_FinalDataSet.csv",replace


///////////////////////////2.2 Combined data without instrument ////////////////

use "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/s.dta",clear
merge m:m  marketid firmid id using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/x.dta"
drop _merge
merge m:m  marketid firmid id using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/w.dta"
drop _merge
export delimited using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/FinalDataSet1.csv",replace


///////////////////////////2.3 Instrument ////////////////

use "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/zd.dta",clear
merge m:m  marketid firmid id using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/zs.dta"
drop _merge
export delimited using "/Users/pengdewendecarmelmarief.zagre/Downloads/Midterm1/FinalDataSet2.csv",replace
