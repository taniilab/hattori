//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2018/02/16 12:13:03
// Design Name: 
// Module Name: top
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module top(
    input wire [3:0]sw,
    input wire [3:0]btn,
    output wire [3:0]led
);


    assign led[0] = (sw[0]) ? btn[0] : ~btn[0];
    assign led[1] = (sw[1]) ? btn[1] : ~btn[1];
    assign led[2] = (sw[2]) ? btn[2] : ~btn[2];
    assign led[3] = (sw[3]) ? btn[3] : ~btn[3];

endmodule