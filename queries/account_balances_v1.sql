SELECT 
    t1."totalBalance"::text AS "demurragedTotalBalance"
    ,t1.account AS "account"
    ,t1."tokenOwner" AS "tokenAddress"
FROM "V_CrcV1_BalancesByAccountAndToken" t1